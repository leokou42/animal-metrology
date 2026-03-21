"""Monocular depth estimation service using Depth Anything V2.

=== Responsibility ===
Estimate per-pixel relative depth from a single RGB image.
Used by the measurement service to correct inter-animal distances
for perspective distortion (animals at different depths).

=== Why Depth Anything V2 ===
1. State-of-the-art monocular depth estimation — zero-shot, no fine-tuning needed
2. Small variant balances speed and accuracy for CPU inference
3. Outputs relative depth map at original image resolution
4. Available via HuggingFace transformers pipeline (auto-downloads on first use)

=== Data Flow ===
Image (BGR numpy) → PIL conversion → Depth Anything V2 → depth_map (H×W float)

=== Graceful Fallback ===
If torch or transformers is not installed, the service logs a warning
and returns None. The pipeline continues with pixel-only distances.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from app.config import settings

if TYPE_CHECKING:
    from transformers import Pipeline

logger = logging.getLogger(__name__)

# Flag to track if depth estimation is available
_depth_available: bool | None = None


def _check_depth_available() -> bool:
    """Check if torch and transformers are installed."""
    global _depth_available
    if _depth_available is not None:
        return _depth_available
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        _depth_available = True
    except ImportError:
        logger.warning(
            "torch or transformers not installed — depth estimation disabled. "
            "Install with: pip install torch transformers"
        )
        _depth_available = False
    return _depth_available


class DepthEstimationService:
    """Estimate per-pixel depth using Depth Anything V2.

    === Lifecycle ===
    1. __init__: Loads the depth estimation pipeline (auto-downloads model)
    2. estimate_depth(): Returns depth map for an image
    3. Model stays loaded for entire API lifecycle (singleton pattern)

    === Output Format ===
    depth_map is a float32 numpy array with shape (H, W).
    Higher values = farther from camera.
    Values are relative (not metric meters).
    """

    def __init__(self, model_name: str | None = None):
        """Load Depth Anything V2 pipeline.

        Args:
            model_name: HuggingFace model ID. Defaults to settings.depth_model.
        """
        model_id = model_name or settings.depth_model

        logger.info("Loading Depth Anything V2 model: %s", model_id)

        from transformers import pipeline

        self.pipe: Pipeline = pipeline(
            "depth-estimation",
            model=model_id,
            device="cpu",
        )
        logger.info("Depth model loaded successfully")

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate per-pixel depth from a BGR image.

        Args:
            image: BGR numpy array (H, W, 3) — OpenCV format

        Returns:
            depth_map: float32 numpy array (H, W).
                       Higher values = farther from camera.
        """
        from PIL import Image

        # Convert BGR (OpenCV) → RGB (PIL)
        rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb)

        h, w = image.shape[:2]

        # Run depth estimation
        result = self.pipe(pil_image)

        # result["depth"] is a PIL Image, convert to numpy
        depth_pil = result["depth"]
        depth_map = np.array(depth_pil, dtype=np.float32)

        # Resize to original image dimensions if needed
        if depth_map.shape[0] != h or depth_map.shape[1] != w:
            import cv2
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        logger.info(
            "Depth estimation complete: shape=%s, range=[%.2f, %.2f]",
            depth_map.shape,
            depth_map.min(),
            depth_map.max(),
        )

        return depth_map


# ============================================================
# Singleton — avoid reloading the model on every request
# ============================================================
_depth_service: DepthEstimationService | None = None


def get_depth_estimation_service() -> DepthEstimationService | None:
    """Get or create the DepthEstimationService singleton.

    Returns None if depth estimation is not available (missing dependencies)
    or disabled in settings.
    """
    global _depth_service

    if not settings.depth_enabled:
        return None

    if not _check_depth_available():
        return None

    if _depth_service is None:
        try:
            _depth_service = DepthEstimationService()
        except Exception:
            logger.warning("Failed to load depth model, disabling depth estimation", exc_info=True)
            return None

    return _depth_service
