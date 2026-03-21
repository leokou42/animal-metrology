"""Eye keypoint detection service using RTMPose with AP-10K pretrained weights.

=== Responsibility ===
Detect left_eye and right_eye keypoints for each segmented animal.
Uses RTMPose-m trained on AP-10K dataset (17 animal keypoints) via ONNX Runtime.

=== Why RTMPose + AP-10K ===
1. AP-10K is the standard benchmark for animal pose estimation — 10,015 images,
   23 species, 17 keypoints including left_eye (idx 0) and right_eye (idx 1).
2. RTMPose-m achieves 72.2 AP on AP-10K while being fast enough for CPU inference.
3. Using ONNX format avoids the heavy mmpose/mmcv dependency chain — only needs
   onnxruntime, which installs as a pure Python wheel.
4. rtmlib handles all preprocessing (affine warp, normalization) and postprocessing
   (SimCC decode, inverse affine) internally.

=== Data Flow ===
SegmentationResult (with bboxes) → RTMPose inference per bbox → list[EyePair]
Each animal's bbox is used as input for top-down pose estimation.
Coordinates are returned in original image pixel space.

=== Model Auto-Download ===
On first use, the ONNX model is downloaded from the OpenMMLab model zoo
to the weights/ directory (~52 MB). Subsequent calls use the cached file.
"""

import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

import numpy as np

from app.config import settings
from app.models.schemas import EyePair, Point
from app.services.segmentation import SegmentationResult

if TYPE_CHECKING:
    from rtmlib import RTMPose

logger = logging.getLogger(__name__)

# AP-10K keypoint indices
LEFT_EYE_IDX = 0
RIGHT_EYE_IDX = 1

# RTMPose AP-10K ONNX model from OpenMMLab model zoo
MODEL_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip"
)
MODEL_INPUT_SIZE = (256, 256)


def _ensure_model(model_path: Path) -> Path:
    """Download the ONNX model if it doesn't exist yet.

    Downloads from OpenMMLab model zoo, extracts the ONNX file from the zip,
    and saves it to the specified path. Same auto-download pattern as
    ultralytics (YOLO weights are downloaded on first use too).
    """
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = model_path.with_suffix(".zip")

    logger.info("Downloading RTMPose AP-10K model to %s ...", model_path)
    urlretrieve(MODEL_URL, str(zip_path))

    # Extract the .onnx file from the nested zip structure
    with zipfile.ZipFile(zip_path, "r") as zf:
        onnx_files = [n for n in zf.namelist() if n.endswith(".onnx")]
        if not onnx_files:
            raise RuntimeError("No .onnx file found in downloaded archive")
        with zf.open(onnx_files[0]) as src, open(model_path, "wb") as dst:
            dst.write(src.read())

    zip_path.unlink()
    logger.info("Model downloaded successfully (%.1f MB)", model_path.stat().st_size / 1024 / 1024)
    return model_path


class EyeDetectionService:
    """Detect animal eye keypoints using RTMPose + AP-10K.

    === Lifecycle ===
    1. __init__: Downloads model (if needed) and loads ONNX via rtmlib
    2. detect_eyes(): Takes SegmentationResult, returns EyePair per animal
    3. detect_and_visualize(): Same as above, plus saves annotated image
    4. Model stays loaded for entire API lifecycle (singleton pattern)

    === Key Design Decisions ===
    - Uses rtmlib.RTMPose for inference — handles affine transforms and
      SimCC decoding internally, so we don't need to reimplement them
    - Accepts pre-computed bboxes from YOLO segmentation (top-down approach)
    - Filters keypoints by confidence threshold — low-confidence eyes set to None
    - Returns coordinates in original image pixel space
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
    ):
        """Load RTMPose ONNX model for animal eye detection.

        Args:
            model_path: Path to the ONNX model file. If not provided,
                        uses settings.eye_model_path. Auto-downloads if missing.
            confidence_threshold: Minimum SimCC score to accept a keypoint.
                                  Below this, the eye is set to None.
        """
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.eye_confidence_threshold
        )

        resolved_path = Path(model_path or settings.eye_model_path)
        _ensure_model(resolved_path)

        logger.info("Loading RTMPose AP-10K model: %s", resolved_path)

        from rtmlib import RTMPose

        self.model: RTMPose = RTMPose(
            onnx_model=str(resolved_path),
            model_input_size=MODEL_INPUT_SIZE,
            backend="onnxruntime",
            device="cpu",
        )
        logger.info("RTMPose model loaded successfully")

    def detect_eyes(self, seg_result: SegmentationResult) -> list[EyePair]:
        """Detect left and right eye keypoints for each segmented animal.

        For each animal in the segmentation result:
        1. Extract its bounding box
        2. Run RTMPose top-down pose estimation within that bbox
        3. Extract left_eye (AP-10K idx 0) and right_eye (idx 1)
        4. Filter by confidence threshold

        Args:
            seg_result: Output from SegmentationService.segment_animals()

        Returns:
            List of EyePair, one per animal in seg_result.animals (same order).
            Eyes below confidence threshold are set to None.
        """
        if seg_result.raw_image is None:
            raise ValueError("SegmentationResult has no raw_image")

        eye_pairs: list[EyePair] = []

        for animal in seg_result.animals:
            try:
                eye_pair = self._detect_single(seg_result.raw_image, animal.bbox)
            except Exception:
                logger.warning(
                    "Eye detection failed for animal #%d (%s), skipping",
                    animal.animal_id,
                    animal.category,
                    exc_info=True,
                )
                eye_pair = EyePair()

            eye_pairs.append(eye_pair)

        detected = sum(
            1 for ep in eye_pairs
            if ep.left_eye is not None or ep.right_eye is not None
        )
        logger.info(
            "Eye detection: %d/%d animals have at least one eye detected",
            detected,
            len(seg_result.animals),
        )

        return eye_pairs

    def detect_and_visualize(
        self,
        seg_result: SegmentationResult,
        image_id: int,
        output_dir: Path | None = None,
    ) -> tuple[list[EyePair], str]:
        """Detect eyes and save an annotated image for visual verification.

        Draws segmentation masks + eye keypoints on the image and saves it
        as i_seg_{image_id}.jpg in the output directory.

        Args:
            seg_result: Output from SegmentationService
            image_id: COCO image ID (used for output filename)
            output_dir: Directory to save the annotated image.
                        Defaults to settings.output_dir.

        Returns:
            Tuple of (eye_pairs, output_path)
        """
        eye_pairs = self.detect_eyes(seg_result)

        # Build eye_data dicts for the visualization function
        eye_data = []
        for animal, eyes in zip(seg_result.animals, eye_pairs):
            eye_data.append({
                "animal_id": animal.animal_id,
                "left_eye": (eyes.left_eye.x, eyes.left_eye.y) if eyes.left_eye else None,
                "right_eye": (eyes.right_eye.x, eyes.right_eye.y) if eyes.right_eye else None,
            })

        from app.utils.visualization import visualize_results

        out_dir = output_dir or settings.output_dir
        output_path = out_dir / f"i_seg_{image_id}.jpg"

        visualize_results(
            seg_result=seg_result,
            eye_data=eye_data,
            output_path=output_path,
        )

        logger.info("Saved eye detection visualization to %s", output_path)
        return eye_pairs, str(output_path)

    def _detect_single(
        self,
        image: np.ndarray,
        bbox: list[float],
    ) -> EyePair:
        """Run pose estimation on a single animal bbox and extract eyes.

        rtmlib.RTMPose handles:
        - Affine transform from bbox to 256x256 model input
        - ImageNet normalization
        - SimCC decode (argmax on x/y logits)
        - Inverse affine to get coordinates in original image space

        Args:
            image: Full image (BGR, numpy array)
            bbox: [x1, y1, x2, y2] bounding box of the animal

        Returns:
            EyePair with left_eye and right_eye (None if below threshold)
        """
        bboxes = np.array([bbox], dtype=np.float32)
        keypoints, scores = self.model(image, bboxes=bboxes)

        # keypoints shape: (1, 17, 2), scores shape: (1, 17)
        kpts = keypoints[0]   # (17, 2)
        confs = scores[0]     # (17,)

        left_eye = None
        if confs[LEFT_EYE_IDX] >= self.confidence_threshold:
            left_eye = Point(
                x=round(float(kpts[LEFT_EYE_IDX][0]), 1),
                y=round(float(kpts[LEFT_EYE_IDX][1]), 1),
            )

        right_eye = None
        if confs[RIGHT_EYE_IDX] >= self.confidence_threshold:
            right_eye = Point(
                x=round(float(kpts[RIGHT_EYE_IDX][0]), 1),
                y=round(float(kpts[RIGHT_EYE_IDX][1]), 1),
            )

        return EyePair(left_eye=left_eye, right_eye=right_eye)


# ============================================================
# Singleton — avoid reloading the ONNX model on every request
# ============================================================
_eye_service: EyeDetectionService | None = None


def get_eye_detection_service() -> EyeDetectionService:
    """Get or create the EyeDetectionService singleton."""
    global _eye_service
    if _eye_service is None:
        _eye_service = EyeDetectionService()
    return _eye_service
