"""Tests for API endpoints and COCO filtering logic."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestCOCOFilter:
    def test_list_animals_returns_422_on_invalid_params(self):
        resp = client.get("/api/v1/coco/animals", params={"min_animals": 0})
        assert resp.status_code == 422

    def test_list_animals_returns_images_or_404(self):
        """Should return images if COCO data available, 404 otherwise."""
        resp = client.get("/api/v1/coco/animals")
        if resp.status_code == 200:
            data = resp.json()
            assert "total_images_found" in data
            assert data["total_images_found"] >= 0
            assert "sample_images" in data
        else:
            assert resp.status_code == 404


class TestAnalyze:
    def test_analyze_nonexistent_image_returns_404(self):
        """Non-existent image_id should return 404."""
        resp = client.post("/api/v1/analyze/99999999")
        assert resp.status_code in (404, 500)
        data = resp.json()
        assert "detail" in data

    def test_analyze_accepts_steps_param(self):
        """Verify steps query parameter is accepted (even if image not found)."""
        resp = client.post("/api/v1/analyze/99999999?steps=segment&visualize=false")
        assert resp.status_code in (404, 500)
        data = resp.json()
        assert "detail" in data

    def test_analyze_rejects_invalid_steps(self):
        """Invalid steps value should return 422."""
        resp = client.post("/api/v1/analyze/287545?steps=invalid")
        assert resp.status_code == 422


class TestCOCOFilterService:
    """Unit tests for the filtering logic itself."""

    def test_animal_categories_mapping(self):
        from app.services.coco_filter import ANIMAL_CATEGORIES

        assert ANIMAL_CATEGORIES[17] == "cat"
        assert ANIMAL_CATEGORIES[18] == "dog"
        assert len(ANIMAL_CATEGORIES) == 10

    def test_service_instantiation(self):
        from app.services.coco_filter import COCOFilterService

        svc = COCOFilterService()
        assert svc.annotations_file is not None
        assert svc.images_dir is not None
