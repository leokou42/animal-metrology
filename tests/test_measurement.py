"""Unit tests for the three-layer distance measurement service."""

import math

import numpy as np

from app.models.schemas import AnimalDetection, EyePair, Point
from app.services.measurement import (
    KNOWN_IOD_CM,
    compute_euclidean_distance,
    compute_inter_distances,
    compute_intra_distances,
    compute_metric_distance,
    measure_all,
    sanity_check,
)


def _make_animal(
    animal_id: int,
    category: str = "cat",
    left_eye: tuple | None = None,
    right_eye: tuple | None = None,
) -> AnimalDetection:
    return AnimalDetection(
        animal_id=animal_id,
        category=category,
        confidence=0.9,
        bbox=[0, 0, 100, 100],
        mask_area=5000,
        eyes=EyePair(
            left_eye=Point(x=left_eye[0], y=left_eye[1]) if left_eye else None,
            right_eye=Point(x=right_eye[0], y=right_eye[1]) if right_eye else None,
        ),
    )


class TestEuclideanDistance:
    def test_3_4_5_triangle(self):
        assert compute_euclidean_distance(Point(x=0, y=0), Point(x=3, y=4)) == 5.0

    def test_same_point(self):
        assert compute_euclidean_distance(Point(x=10, y=20), Point(x=10, y=20)) == 0.0

    def test_horizontal(self):
        assert compute_euclidean_distance(Point(x=0, y=0), Point(x=7, y=0)) == 7.0


class TestMetricDistance:
    def test_same_depth_plane(self):
        """Two points at same depth — metric dist should be proportional to pixel dist."""
        depth_map = np.full((100, 100), 5.0, dtype=np.float32)
        d, z1, z2 = compute_metric_distance(
            Point(x=40, y=50), Point(x=60, y=50),
            depth_map, focal_length=500.0, cx=50.0, cy=50.0,
        )
        assert d > 0
        assert z1 == 5.0
        assert z2 == 5.0

    def test_different_depths(self):
        """Points at different depths should have non-zero Z component."""
        depth_map = np.full((100, 100), 5.0, dtype=np.float32)
        depth_map[50, 60] = 10.0
        d, z1, z2 = compute_metric_distance(
            Point(x=40, y=50), Point(x=60, y=50),
            depth_map, focal_length=500.0, cx=50.0, cy=50.0,
        )
        assert z1 == 5.0
        assert z2 == 10.0
        assert d > 0


class TestSanityCheck:
    def test_cat_pass(self):
        assert sanity_check("cat", 0.055) == "PASS"  # 5.5cm

    def test_cat_warning(self):
        assert sanity_check("cat", 0.08) == "WARNING"  # 8cm, outside 5-6 but within tolerance

    def test_cat_fail(self):
        assert sanity_check("cat", 0.5) == "FAIL"  # 50cm, clearly wrong

    def test_unknown_category(self):
        assert sanity_check("platypus", 0.05) is None

    def test_known_categories_exist(self):
        for cat in ["cat", "dog", "giraffe", "sheep"]:
            assert cat in KNOWN_IOD_CM


class TestIntraDistances:
    def test_pixel_only(self):
        animals = [_make_animal(0, left_eye=(0, 0), right_eye=(3, 4))]
        result = compute_intra_distances(animals)
        assert len(result) == 1
        assert result[0].pixel_distance == 5.0
        assert result[0].metric_distance_m is None
        assert result[0].sanity_check_result is None

    def test_with_metric_depth(self):
        animals = [_make_animal(0, "cat", left_eye=(40, 50), right_eye=(60, 50))]
        depth_map = np.full((100, 100), 5.0, dtype=np.float32)
        result = compute_intra_distances(
            animals, depth_map=depth_map, focal_length=500.0, image_size=(100, 100)
        )
        assert len(result) == 1
        assert result[0].metric_distance_m is not None
        assert result[0].sanity_check_result is not None

    def test_missing_eye_skipped(self):
        animals = [_make_animal(0, left_eye=None, right_eye=(3, 4))]
        assert compute_intra_distances(animals) == []

    def test_empty_list(self):
        assert compute_intra_distances([]) == []


class TestInterDistances:
    def test_two_animals(self):
        animals = [
            _make_animal(0, right_eye=(10, 0)),
            _make_animal(1, right_eye=(10, 30)),
        ]
        result = compute_inter_distances(animals)
        assert len(result) == 1
        assert result[0].pixel_distance == 30.0
        assert result[0].metric_distance_m is None

    def test_three_animals_three_pairs(self):
        animals = [
            _make_animal(0, right_eye=(0, 0)),
            _make_animal(1, right_eye=(3, 4)),
            _make_animal(2, right_eye=(0, 10)),
        ]
        assert len(compute_inter_distances(animals)) == 3

    def test_no_right_eye_excluded(self):
        animals = [
            _make_animal(0, right_eye=(0, 0)),
            _make_animal(1, left_eye=(5, 5)),
        ]
        assert compute_inter_distances(animals) == []

    def test_with_metric_depth(self):
        animals = [
            _make_animal(0, right_eye=(30, 50)),
            _make_animal(1, right_eye=(70, 50)),
        ]
        depth_map = np.full((100, 100), 5.0, dtype=np.float32)
        result = compute_inter_distances(
            animals, depth_map=depth_map, focal_length=500.0, image_size=(100, 100)
        )
        assert len(result) == 1
        assert result[0].metric_distance_m is not None


class TestMeasureAll:
    def test_without_depth(self):
        animals = [
            _make_animal(0, "dog", left_eye=(0, 0), right_eye=(3, 4)),
            _make_animal(1, "cat", left_eye=(10, 10), right_eye=(13, 14)),
        ]
        intra, inter = measure_all(animals)
        assert len(intra) == 2
        assert len(inter) == 1
        assert intra[0].metric_distance_m is None

    def test_with_depth(self):
        animals = [
            _make_animal(0, "dog", left_eye=(20, 30), right_eye=(40, 30)),
            _make_animal(1, "cat", left_eye=(60, 70), right_eye=(80, 70)),
        ]
        depth_map = np.full((100, 100), 5.0, dtype=np.float32)
        intra, inter = measure_all(
            animals, depth_map=depth_map, focal_length=500.0, image_size=(100, 100)
        )
        assert len(intra) == 2
        assert len(inter) == 1
        assert intra[0].metric_distance_m is not None
        assert inter[0].metric_distance_m is not None
