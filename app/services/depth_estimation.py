"""Monocular depth estimation service with metric depth support.

=== Responsibility ===
Estimate per-pixel depth from a single RGB image. Provides both the depth map
and (when available) the estimated focal length for 3D metric distance computation.

=== Model Priority ===
1. Apple Depth Pro — outputs metric depth (meters) + focal length (pixels)
   Best choice: enables true 3D distance calculation without camera calibration.
2. Depth Anything V2 (fallback) — outputs relative depth (no units)
   Used when Depth Pro is not installed or fails to load.

=== Data Flow ===
Image (BGR numpy) → Model inference → DepthResult(depth_map, focal_length, is_metric)

=== Graceful Fallback ===
If neither model is available, the service returns None and the pipeline
continues with pixel-only distances.
"""

import logging
from dataclasses import dataclass

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Output from depth estimation.

    Attributes:
        depth_map: float32 array (H, W). Per-pixel depth values.
        focal_length_px: Estimated focal length in pixels (None for relative depth).
        is_metric: True if depth values are in meters (Depth Pro),
                   False if relative/unitless (DA V2).
    """
    depth_map: np.ndarray
    focal_length_px: float | None
    is_metric: bool


def _try_load_depth_pro() -> object | None:
    """Try to load Apple Depth Pro. Returns (model, transform) or None."""
    try:
        import depth_pro
        import torch
        from depth_pro.depth_pro import DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT

        checkpoint = settings.depth_pro_checkpoint
        config = DepthProConfig(
            patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
            image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
            decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
            checkpoint_uri=checkpoint,
            fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
            use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
        )

        model, transform = depth_pro.create_model_and_transforms(
            config=config, device=torch.device("cpu")
        )
        model.eval()
        logger.info("Loaded Apple Depth Pro (metric depth)")
        return model, transform
    except Exception as e:
        logger.info("Depth Pro not available (%s), trying fallback", e)
        return None


def _try_load_da_v2() -> object | None:
    """Try to load Depth Anything V2. Returns pipeline or None."""
    try:
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline(
            "depth-estimation",
            model=settings.depth_model,
            device="cpu",
        )
        logger.info("Loaded Depth Anything V2 (relative depth)")
        return pipe
    except Exception as e:
        logger.info("Depth Anything V2 not available (%s)", e)
        return None


class DepthEstimationService:
    """Estimate per-pixel depth with automatic model selection.

    Tries Depth Pro first (metric depth in meters + focal length).
    Falls back to Depth Anything V2 (relative depth, no focal length).
    """

    def __init__(self):
        self._depth_pro = None
        self._da_v2 = None
        self._backend = None

        # Try Depth Pro first
        result = _try_load_depth_pro()
        if result is not None:
            self._depth_pro = result  # (model, transform)
            self._backend = "depth_pro"
        else:
            logger.warning(
                "Depth Pro not available — install with: "
                "pip install depth-pro. Falling back to DA V2 (fast mode)."
            )

        # Also try DA V2 (used when depth_pro=fast, or as fallback)
        result = _try_load_da_v2()
        if result is not None:
            self._da_v2 = result
            if self._backend is None:
                self._backend = "da_v2"

        if self._backend is None:
            raise RuntimeError("No depth estimation model available")

    @property
    def backend_name(self) -> str:
        return self._backend or "none"

    def estimate_depth(
        self, image: np.ndarray, prefer_metric: bool = True,
    ) -> DepthResult:
        """Estimate depth from a BGR image.

        Args:
            image: BGR numpy array (H, W, 3)
            prefer_metric: If True, use Depth Pro (metric, slow).
                           If False, use DA V2 even if Depth Pro is available.

        Returns:
            DepthResult with depth_map, optional focal_length, and is_metric flag
        """
        # If caller wants fast mode and DA V2 is available, use it
        if not prefer_metric and self._da_v2 is not None:
            return self._run_da_v2(image)

        # Otherwise use the best available backend
        if self._backend == "depth_pro" and prefer_metric:
            return self._run_depth_pro(image)
        elif self._backend == "da_v2" or self._da_v2 is not None:
            return self._run_da_v2(image)
        elif self._backend == "depth_pro":
            return self._run_depth_pro(image)
        else:
            raise RuntimeError("No depth backend loaded")

    def _run_depth_pro(self, image: np.ndarray) -> DepthResult:
        """Run Apple Depth Pro inference."""
        import torch
        import depth_pro
        from PIL import Image as PILImage
        import tempfile

        model, transform = self._depth_pro
        h, w = image.shape[:2]

        # Depth Pro's load_rgb expects a file path, so save temporarily
        rgb = image[:, :, ::-1]
        pil_img = PILImage.fromarray(rgb)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_img.save(tmp.name)
            img_tensor, _, f_px = depth_pro.load_rgb(tmp.name)

        img_tensor = transform(img_tensor)

        with torch.no_grad():
            prediction = model.infer(img_tensor, f_px=f_px)

        depth_map = prediction["depth"].cpu().numpy()
        focal_length = prediction["focallength_px"].item()

        # Resize to original image dimensions if needed
        if depth_map.shape[0] != h or depth_map.shape[1] != w:
            import cv2
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        logger.info(
            "Depth Pro: shape=%s, range=[%.2f, %.2f]m, focal=%.1fpx",
            depth_map.shape, depth_map.min(), depth_map.max(), focal_length,
        )

        return DepthResult(
            depth_map=depth_map.astype(np.float32),
            focal_length_px=focal_length,
            is_metric=True,
        )

    def _run_da_v2(self, image: np.ndarray) -> DepthResult:
        """Run Depth Anything V2 inference (relative depth)."""
        from PIL import Image as PILImage

        rgb = image[:, :, ::-1]
        pil_image = PILImage.fromarray(rgb)
        h, w = image.shape[:2]

        result = self._da_v2(pil_image)
        depth_pil = result["depth"]
        depth_map = np.array(depth_pil, dtype=np.float32)

        if depth_map.shape[0] != h or depth_map.shape[1] != w:
            import cv2
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        logger.info(
            "DA V2: shape=%s, range=[%.2f, %.2f] (relative)",
            depth_map.shape, depth_map.min(), depth_map.max(),
        )

        return DepthResult(
            depth_map=depth_map,
            focal_length_px=None,
            is_metric=False,
        )



# ============================================================
# Singleton
# ============================================================
_depth_service: DepthEstimationService | None = None
_depth_init_attempted: bool = False


def get_depth_estimation_service(
    prefer_metric: bool = True,
) -> DepthEstimationService | None:
    """Get or create the DepthEstimationService singleton.

    Args:
        prefer_metric: Passed through to estimate_depth() calls.
            Not used here — the singleton loads all available backends.

    Returns None if depth estimation is disabled or no model available.
    """
    global _depth_service, _depth_init_attempted

    if not settings.depth_enabled:
        return None

    if _depth_init_attempted:
        return _depth_service

    _depth_init_attempted = True
    try:
        _depth_service = DepthEstimationService()
    except Exception:
        logger.warning("Depth estimation unavailable", exc_info=True)
        _depth_service = None

    return _depth_service
