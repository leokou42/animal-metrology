"""Unit tests for the distance measurement service."""

import math

import numpy as np

from app.models.schemas import AnimalDetection, EyePair, Point
from app.services.measurement import (
    compute_euclidean_distance,
    compute_depth_corrected_inter_distances,
    compute_inter_distances,
    compute_intra_distances,
    measure_all,
)


def _make_animal(
    animal_id: int,
    category: str = "cat",
    left_eye: tuple | None = None,
    right_eye: tuple | None = None,
) -> AnimalDetection:
    """Helper to build AnimalDetection with minimal boilerplate."""
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
        d = compute_euclidean_distance(Point(x=0, y=0), Point(x=3, y=4))
        assert d == 5.0

    def test_same_point_returns_zero(self):
        d = compute_euclidean_distance(Point(x=10, y=20), Point(x=10, y=20))
        assert d == 0.0

    def test_horizontal_distance(self):
        d = compute_euclidean_distance(Point(x=0, y=0), Point(x=7, y=0))
        assert d == 7.0

    def test_diagonal(self):
        d = compute_euclidean_distance(Point(x=0, y=0), Point(x=1, y=1))
        assert abs(d - math.sqrt(2)) < 1e-10


class TestIntraDistances:
    def test_both_eyes_present(self):
        animals = [_make_animal(0, left_eye=(0, 0), right_eye=(3, 4))]
        result = compute_intra_distances(animals)
        assert len(result) == 1
        assert result[0].distance_px == 5.0

    def test_missing_eye_skipped(self):
        animals = [_make_animal(0, left_eye=None, right_eye=(3, 4))]
        assert compute_intra_distances(animals) == []

    def test_no_eyes_skipped(self):
        animals = [_make_animal(0)]
        assert compute_intra_distances(animals) == []

    def test_empty_list(self):
        assert compute_intra_distances([]) == []

    def test_multiple_animals(self):
        animals = [
            _make_animal(0, left_eye=(0, 0), right_eye=(3, 4)),
            _make_animal(1, left_eye=(10, 10), right_eye=(10, 20)),
        ]
        result = compute_intra_distances(animals)
        assert len(result) == 2
        assert result[0].distance_px == 5.0
        assert result[1].distance_px == 10.0


class TestInterDistances:
    def test_two_animals_one_pair(self):
        animals = [
            _make_animal(0, right_eye=(10, 0)),
            _make_animal(1, right_eye=(10, 30)),
        ]
        result = compute_inter_distances(animals)
        assert len(result) == 1
        assert result[0].distance_px == 30.0

    def test_three_animals_three_pairs(self):
        animals = [
            _make_animal(0, right_eye=(0, 0)),
            _make_animal(1, right_eye=(3, 4)),
            _make_animal(2, right_eye=(0, 10)),
        ]
        result = compute_inter_distances(animals)
        assert len(result) == 3

    def test_animal_without_right_eye_excluded(self):
        animals = [
            _make_animal(0, right_eye=(0, 0)),
            _make_animal(1, left_eye=(5, 5)),  # no right eye
            _make_animal(2, right_eye=(10, 0)),
        ]
        result = compute_inter_distances(animals)
        assert len(result) == 1
        assert result[0].animal_a_id == 0
        assert result[0].animal_b_id == 2

    def test_empty_list(self):
        assert compute_inter_distances([]) == []

    def test_single_animal_no_pairs(self):
        animals = [_make_animal(0, right_eye=(5, 5))]
        assert compute_inter_distances(animals) == []


class TestDepthCorrectedDistances:
    def test_same_depth_no_correction(self):
        """If both animals are at the same depth, corrected == pixel distance."""
        animals = [
            _make_animal(0, right_eye=(10, 10)),
            _make_animal(1, right_eye=(20, 10)),
        ]
        # Uniform depth map (all same value)
        depth_map = np.full((100, 100), 50.0, dtype=np.float32)
        result = compute_depth_corrected_inter_distances(animals, depth_map)
        assert len(result) == 1
        assert result[0].pixel_distance == result[0].depth_corrected_distance

    def test_different_depths_corrects_upward(self):
        """Animal farther away should increase corrected distance."""
        animals = [
            _make_animal(0, right_eye=(10, 10)),
            _make_animal(1, right_eye=(20, 10)),
        ]
        depth_map = np.full((100, 100), 50.0, dtype=np.float32)
        # Animal 1 is farther (higher depth value)
        depth_map[10, 20] = 100.0
        result = compute_depth_corrected_inter_distances(animals, depth_map)
        assert len(result) == 1
        assert result[0].depth_corrected_distance > result[0].pixel_distance

    def test_missing_right_eye_excluded(self):
        animals = [
            _make_animal(0, right_eye=(10, 10)),
            _make_animal(1, left_eye=(5, 5)),  # no right eye
        ]
        depth_map = np.full((100, 100), 50.0, dtype=np.float32)
        assert compute_depth_corrected_inter_distances(animals, depth_map) == []


class TestMeasureAll:
    def test_without_depth(self):
        animals = [
            _make_animal(0, "dog", left_eye=(0, 0), right_eye=(3, 4)),
            _make_animal(1, "cat", left_eye=(10, 10), right_eye=(13, 14)),
        ]
        intra, inter, depth_corr = measure_all(animals)
        assert len(intra) == 2
        assert len(inter) == 1
        assert depth_corr is None

    def test_with_depth(self):
        animals = [
            _make_animal(0, "dog", left_eye=(0, 0), right_eye=(3, 4)),
            _make_animal(1, "cat", left_eye=(10, 10), right_eye=(13, 14)),
        ]
        depth_map = np.full((100, 100), 50.0, dtype=np.float32)
        intra, inter, depth_corr = measure_all(animals, depth_map=depth_map)
        assert len(intra) == 2
        assert len(inter) == 1
        assert depth_corr is not None
        assert len(depth_corr) == 1
