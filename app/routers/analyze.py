"""API endpoints for the animal eye metrology pipeline.

Endpoints:
  GET  /api/v1/coco/animals          — Browse COCO images with multiple animals
  POST /api/v1/analyze/{image_id}    — Run pipeline on a COCO image
  POST /api/v1/analyze/upload        — Run pipeline on an uploaded image

Query parameters:
  steps:     segment | eyes | full (default: full)
  visualize: true | false (default: true)
"""

import logging
import tempfile
from enum import Enum
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from app.config import settings
from app.models.schemas import COCOFilterResult, MeasurementResult
from app.services.coco_filter import get_coco_filter_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


class PipelineSteps(str, Enum):
    segment = "segment"  # Segmentation only
    eyes = "eyes"        # Segmentation + eye detection
    full = "full"        # Segmentation + eyes + distance measurement


class DepthMode(str, Enum):
    none = "none"        # Pixel distance only, no depth model
    fast = "fast"        # Depth Anything V2 (relative depth, ~2s)
    metric = "metric"    # Depth Pro (metric depth in meters, ~30-60s)


# ============================================================
# COCO browsing
# ============================================================

@router.get("/coco/animals", response_model=COCOFilterResult)
async def list_multi_animal_images(
    min_animals: int = Query(default=2, ge=2, description="Minimum animal count"),
    max_animals: int | None = Query(default=None, ge=2, description="Maximum animal count"),
    category: list[str] | None = Query(default=None, description="Filter by animal type, e.g. cat, dog, elephant"),
    max_results: int = Query(default=20, ge=1, le=100, description="Max images to return"),
):
    """List COCO images containing multiple animals."""
    try:
        svc = get_coco_filter_service()
        images = svc.filter_multi_animal_images(
            min_animals=min_animals,
            max_animals=max_animals,
            category_filter=category,
            max_results=max_results,
        )
        return COCOFilterResult(
            total_images_found=len(images),
            sample_images=images,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"COCO annotations not found. Check COCO_ANNOTATIONS_FILE path. {e}",
        )
    except Exception as e:
        logger.exception("Failed to filter COCO images")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Analysis pipeline
# ============================================================

def _run_pipeline(
    image_path: str,
    image_id: int,
    image_file: str,
    image_width: int,
    image_height: int,
    steps: PipelineSteps,
    visualize: bool,
    depth_pro: DepthMode = DepthMode.metric,
) -> dict:
    """Shared pipeline logic for both COCO and upload endpoints.

    Runs pipeline stages based on `steps` parameter:
      segment → eyes → full
    Each stage includes all previous stages.

    depth_pro controls which depth model to use (if steps==full):
      none   → pixel distances only
      fast   → Depth Anything V2 (relative depth, ~2s)
      metric → Depth Pro (metric depth in meters, ~30-60s)
    """
    from app.services.segmentation import get_segmentation_service
    from app.models.schemas import AnimalDetection, EyePair

    # --- Stage 1: Segmentation (always runs) ---
    seg_svc = get_segmentation_service()
    seg_result = seg_svc.segment_animals(image_path)

    # Default: no eyes, no distances
    eye_pairs = [EyePair() for _ in seg_result.animals]
    intra_distances = []
    inter_distances = []

    # --- Stage 2: Eye detection (if steps >= eyes) ---
    if steps in (PipelineSteps.eyes, PipelineSteps.full):
        from app.services.eye_detection import get_eye_detection_service

        eye_svc = get_eye_detection_service()
        eye_pairs = eye_svc.detect_eyes(seg_result)

    # Build AnimalDetection list (needed by measurement and response)
    animals = [
        AnimalDetection(
            animal_id=a.animal_id,
            category=a.category,
            confidence=a.confidence,
            bbox=a.bbox,
            mask_area=a.mask_area,
            eyes=ep,
        )
        for a, ep in zip(seg_result.animals, eye_pairs)
    ]

    # --- Stage 3: Distance measurement (if steps == full) ---
    if steps == PipelineSteps.full:
        from app.services.measurement import measure_all

        # Depth estimation based on depth mode
        depth_map = None
        focal_length = None
        if depth_pro != DepthMode.none:
            try:
                from app.services.depth_estimation import get_depth_estimation_service

                depth_svc = get_depth_estimation_service()
                if depth_svc is not None and seg_result.raw_image is not None:
                    depth_result = depth_svc.estimate_depth(
                        seg_result.raw_image,
                        prefer_metric=(depth_pro == DepthMode.metric),
                    )
                    depth_map = depth_result.depth_map
                    if depth_result.is_metric:
                        focal_length = depth_result.focal_length_px
            except Exception:
                logger.warning("Depth estimation failed, continuing without it", exc_info=True)

        intra_distances, inter_distances = measure_all(
            animals,
            depth_map=depth_map,
            focal_length=focal_length,
            image_size=seg_result.image_hw,
        )

    # --- Stage 4: Visualization (if requested) ---
    annotated_image_path = None
    if visualize:
        from app.utils.visualization import visualize_results

        output_filename = f"analyze_{image_id}.jpg"
        output_path = settings.output_dir / output_filename
        annotated_image_path = f"/outputs/{output_filename}"

        # Build viz-compatible dicts
        eye_data = None
        intra_dicts = None
        inter_dicts = None

        if steps in (PipelineSteps.eyes, PipelineSteps.full):
            eye_data = [
                {
                    "animal_id": a.animal_id,
                    "left_eye": (ep.left_eye.x, ep.left_eye.y) if ep.left_eye else None,
                    "right_eye": (ep.right_eye.x, ep.right_eye.y) if ep.right_eye else None,
                }
                for a, ep in zip(seg_result.animals, eye_pairs)
            ]

        if steps == PipelineSteps.full:
            intra_dicts = [
                {
                    "animal_id": d.animal_id,
                    "category": d.category,
                    "left_eye": (d.left_eye.x, d.left_eye.y),
                    "right_eye": (d.right_eye.x, d.right_eye.y),
                    "distance_px": d.pixel_distance,
                    "depth_corrected_px": d.depth_corrected_px,
                    "metric_distance_m": d.metric_distance_m,
                    "sanity_check_result": d.sanity_check_result,
                }
                for d in intra_distances
            ]
            inter_dicts = [
                {
                    "animal_a_id": d.animal_a_id,
                    "animal_b_id": d.animal_b_id,
                    "eye_a": (d.eye_a.x, d.eye_a.y),
                    "eye_b": (d.eye_b.x, d.eye_b.y),
                    "distance_px": d.pixel_distance,
                    "depth_corrected_px": d.depth_corrected_px,
                    "metric_distance_m": d.metric_distance_m,
                }
                for d in inter_distances
            ]

        visualize_results(
            seg_result=seg_result,
            eye_data=eye_data,
            intra_distances=intra_dicts,
            inter_distances=inter_dicts,
            output_path=output_path,
        )

    return MeasurementResult(
        image_id=image_id,
        image_file=image_file,
        image_width=image_width,
        image_height=image_height,
        animals=animals,
        intra_distances=intra_distances,
        inter_distances=inter_distances,
        annotated_image_path=annotated_image_path,
    ).model_dump()


@router.post("/analyze/upload", response_model=MeasurementResult)
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    steps: PipelineSteps = Query(
        default=PipelineSteps.full,
        description="Pipeline depth: segment, eyes, or full",
    ),
    visualize: bool = Query(
        default=True,
        description="Generate annotated image in outputs/",
    ),
    depth_pro: DepthMode = Query(
        default=DepthMode.metric,
        description="Depth mode: none (pixel only), fast (DA V2 ~2s), metric (Depth Pro ~30-60s)",
    ),
):
    """Run analysis pipeline on an uploaded image.

    Accepts JPG/PNG. No COCO dataset required.
    Same `steps`, `visualize`, and `depth_pro` parameters as the COCO endpoint.
    """
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    tmp_path = None
    try:
        suffix = ".jpg" if "jpeg" in file.content_type else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        import cv2
        img = cv2.imread(tmp_path)
        if img is None:
            raise ValueError(f"Failed to read uploaded image: {file.filename}")
        h, w = img.shape[:2]

        stem = Path(file.filename).stem if file.filename else "uploaded"

        return _run_pipeline(
            image_path=tmp_path,
            image_id=hash(stem) % 10**8,
            image_file=file.filename or "uploaded.jpg",
            image_width=w,
            image_height=h,
            steps=steps,
            visualize=visualize,
            depth_pro=depth_pro,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis failed for uploaded file %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@router.post("/analyze/{image_id}", response_model=MeasurementResult)
async def analyze_image(
    image_id: int,
    steps: PipelineSteps = Query(
        default=PipelineSteps.full,
        description="Pipeline depth: segment, eyes, or full",
    ),
    visualize: bool = Query(
        default=True,
        description="Generate annotated image in outputs/",
    ),
    depth_pro: DepthMode = Query(
        default=DepthMode.metric,
        description="Depth mode: none (pixel only), fast (DA V2 ~2s), metric (Depth Pro ~30-60s)",
    ),
):
    """Run analysis pipeline on a COCO image.

    Control what to compute with `steps`:
      - segment: animal contours + bounding boxes only
      - eyes: + eye keypoint coordinates
      - full: + inter-ocular and inter-animal distances

    Control depth model with `depth_pro`:
      - none: pixel distances only (fastest)
      - fast: Depth Anything V2 relative depth (~2s)
      - metric: Depth Pro metric depth in meters (~30-60s)

    Control output with `visualize`:
      - true: also save annotated image to /outputs/analyze_{image_id}.jpg
      - false: JSON response only
    """
    try:
        coco_svc = get_coco_filter_service()
        img_data = coco_svc.get_image_annotations(image_id)
        img_info = img_data["image_info"]

        image_path = Path(coco_svc.images_dir) / img_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return _run_pipeline(
            image_path=str(image_path),
            image_id=image_id,
            image_file=img_info["file_name"],
            image_width=img_info["width"],
            image_height=img_info["height"],
            steps=steps,
            visualize=visualize,
            depth_pro=depth_pro,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed for image %d", image_id)
        raise HTTPException(status_code=500, detail=str(e))
