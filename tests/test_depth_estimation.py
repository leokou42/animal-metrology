"""Unit tests for depth estimation device selection."""

import pytest
import torch

from app.config import settings
from app.services import depth_estimation


class TestDepthDeviceResolution:
    def test_auto_uses_cpu_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr(settings, "depth_device", "auto")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        device = depth_estimation._resolve_torch_device()

        assert device.type == "cpu"

    def test_auto_prefers_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(settings, "depth_device", "auto")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        device = depth_estimation._resolve_torch_device()

        assert device.type == "cuda"

    def test_explicit_cuda_requires_available_runtime(self, monkeypatch):
        monkeypatch.setattr(settings, "depth_device", "cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError, match="requested CUDA"):
            depth_estimation._resolve_torch_device()

    def test_explicit_cuda_index_must_exist(self, monkeypatch):
        monkeypatch.setattr(settings, "depth_device", "cuda:1")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

        with pytest.raises(RuntimeError, match="unavailable GPU index 1"):
            depth_estimation._resolve_torch_device()

    def test_invalid_setting_is_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "depth_device", "tpu")

        with pytest.raises(ValueError, match="Unsupported DEPTH_DEVICE"):
            depth_estimation._resolve_torch_device()


class TestTransformersDeviceMapping:
    def test_cpu_maps_to_negative_one(self):
        assert depth_estimation._resolve_transformers_device(torch.device("cpu")) == -1

    def test_cuda_maps_to_index_zero_by_default(self):
        assert depth_estimation._resolve_transformers_device(torch.device("cuda")) == 0

    def test_cuda_maps_to_requested_index(self):
        assert depth_estimation._resolve_transformers_device(torch.device("cuda:2")) == 2
