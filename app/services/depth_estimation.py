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
from pathlib import Path

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


def _resolve_torch_device():
    """Resolve the runtime device for depth models from settings."""
    import torch

    requested = settings.depth_device.strip().lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"DEPTH_DEVICE={settings.depth_device!r} requested CUDA, "
                "but torch.cuda.is_available() is False."
            )

        device = torch.device(requested)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"DEPTH_DEVICE={settings.depth_device!r} requested unavailable GPU "
                f"index {device.index}. Available CUDA devices: {torch.cuda.device_count()}."
            )
        return device

    raise ValueError(
        f"Unsupported DEPTH_DEVICE={settings.depth_device!r}. "
        "Use one of: auto, cpu, cuda, cuda:N."
    )


def _resolve_transformers_device(torch_device) -> int:
    """Translate a torch device to the Hugging Face pipeline device format."""
    if torch_device.type == "cuda":
        return torch_device.index if torch_device.index is not None else 0
    return -1


def _try_load_depth_pro() -> object | None:
    """Try to load Apple Depth Pro. Returns (model, transform) or None."""
    device = _resolve_torch_device()
    try:
        import depth_pro
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
            config=config, device=device
        )
        model.eval()
        logger.info("Loaded Apple Depth Pro (metric depth) on %s", device)
        return model, transform, device
    except Exception as e:
        logger.info("Depth Pro not available (%s), trying fallback", e)
        return None


def _try_load_da_v2() -> object | None:
    """Try to load Depth Anything V2. Returns pipeline or None."""
    device = _resolve_torch_device()
    try:
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline(
            "depth-estimation",
            model=settings.depth_model,
            device=_resolve_transformers_device(device),
        )
        logger.info("Loaded Depth Anything V2 (relative depth) on %s", device)
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
        if not prefer_metric:
            # Fast mode: DA V2 only
            if self._da_v2 is not None:
                return self._run_da_v2(image)
            # DA V2 not available — fall back to Depth Pro with warning
            if self._backend == "depth_pro":
                logger.warning(
                    "DA V2 not available for fast mode, falling back to Depth Pro (slower)"
                )
                return self._run_depth_pro(image)
            raise RuntimeError("No fast depth backend available (DA V2 not installed)")

        # Metric mode: prefer Depth Pro, fall back to DA V2
        if self._backend == "depth_pro":
            return self._run_depth_pro(image)
        if self._da_v2 is not None:
            return self._run_da_v2(image)
        raise RuntimeError("No depth backend loaded")

    def _run_depth_pro(self, image: np.ndarray) -> DepthResult:
        """Run Apple Depth Pro inference."""
        import torch
        import depth_pro
        from PIL import Image as PILImage
        import tempfile

        model, transform, device = self._depth_pro
        h, w = image.shape[:2]

        # Depth Pro's load_rgb expects a file path, so save temporarily
        rgb = image[:, :, ::-1]
        pil_img = PILImage.fromarray(rgb)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                pil_img.save(tmp_path)
            img_tensor, _, f_px = depth_pro.load_rgb(tmp_path)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        img_tensor = transform(img_tensor).to(device)
        if isinstance(f_px, torch.Tensor):
            f_px = f_px.to(device)

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

    try:
        _depth_service = DepthEstimationService()
        _depth_init_attempted = True  # Only mark after success
    except Exception:
        logger.warning("Depth estimation unavailable, will retry next call", exc_info=True)
        _depth_service = None

    return _depth_service
