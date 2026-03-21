"""Distance measurement service for animal eye metrology.

=== Responsibility ===
Compute distances between eye keypoints at two levels:
  Layer 1 (2D Pixel): Euclidean distance in image pixel space
  Layer 2 (Depth-Aware): Perspective-corrected distance using monocular depth

=== Data Flow ===
list[AnimalDetection] + optional depth_map → distance calculations → result lists

=== Units ===
Layer 1: pixels — no physical meaning, affected by perspective
Layer 2: depth-corrected pixels — compensates for animals at different distances
         from camera, but still not metric (cm/mm)
"""

import logging
import math
from itertools import combinations

import numpy as np

from app.models.schemas import (
    AnimalDetection,
    DepthCorrectedInterDistance,
    InterAnimalDistance,
    IntraAnimalDistance,
    Point,
)

logger = logging.getLogger(__name__)


def compute_euclidean_distance(p1: Point, p2: Point) -> float:
    """Compute Euclidean distance between two points in pixel space.

    Formula: d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    """
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def compute_intra_distances(
    animals: list[AnimalDetection],
) -> list[IntraAnimalDistance]:
    """Compute the distance between left and right eyes for each animal.

    Animals missing either eye are skipped silently.
    """
    results: list[IntraAnimalDistance] = []

    for animal in animals:
        if animal.eyes.left_eye is None or animal.eyes.right_eye is None:
            continue

        distance = compute_euclidean_distance(
            animal.eyes.left_eye, animal.eyes.right_eye
        )
        results.append(
            IntraAnimalDistance(
                animal_id=animal.animal_id,
                category=animal.category,
                left_eye=animal.eyes.left_eye,
                right_eye=animal.eyes.right_eye,
                distance_px=round(distance, 2),
            )
        )

    return results


def compute_inter_distances(
    animals: list[AnimalDetection],
) -> list[InterAnimalDistance]:
    """Compute pairwise distances between right eyes of all animals.

    Uses itertools.combinations to generate all unique pairs.
    Animals without a right_eye are excluded before pairing.
    """
    eligible = [a for a in animals if a.eyes.right_eye is not None]

    results: list[InterAnimalDistance] = []

    for a, b in combinations(eligible, 2):
        distance = compute_euclidean_distance(a.eyes.right_eye, b.eyes.right_eye)
        results.append(
            InterAnimalDistance(
                animal_a_id=a.animal_id,
                animal_a_category=a.category,
                animal_b_id=b.animal_id,
                animal_b_category=b.category,
                eye_a=a.eyes.right_eye,
                eye_b=b.eyes.right_eye,
                distance_px=round(distance, 2),
            )
        )

    return results


def compute_depth_corrected_inter_distances(
    animals: list[AnimalDetection],
    depth_map: np.ndarray,
) -> list[DepthCorrectedInterDistance]:
    """Compute depth-corrected pairwise distances between right eyes.

    Depth Anything V2 outputs relative depth (not metric).
    Animals farther from camera have larger depth values and appear smaller,
    so their pixel distances are compressed. We correct by scaling:

        corrected = pixel_distance * avg_depth / min(depth_a, depth_b)

    This normalizes distances as if both animals were at the closer depth,
    giving a fairer comparison than raw pixel distances.

    Args:
        animals: list of AnimalDetection with eye coordinates
        depth_map: float32 array (H, W) from DepthEstimationService

    Returns:
        list of DepthCorrectedInterDistance for all valid pairs
    """
    eligible = [a for a in animals if a.eyes.right_eye is not None]
    h, w = depth_map.shape[:2]

    results: list[DepthCorrectedInterDistance] = []

    for a, b in combinations(eligible, 2):
        eye_a = a.eyes.right_eye
        eye_b = b.eyes.right_eye

        pixel_dist = compute_euclidean_distance(eye_a, eye_b)

        # Look up depth at eye coordinates (clamp to image bounds)
        ax, ay = int(min(max(eye_a.x, 0), w - 1)), int(min(max(eye_a.y, 0), h - 1))
        bx, by = int(min(max(eye_b.x, 0), w - 1)), int(min(max(eye_b.y, 0), h - 1))

        depth_a = float(depth_map[ay, ax])
        depth_b = float(depth_map[by, bx])

        # Avoid division by zero
        min_depth = min(depth_a, depth_b)
        if min_depth <= 0:
            logger.warning(
                "Skipping depth correction for pair (%d, %d): min_depth=%.2f",
                a.animal_id, b.animal_id, min_depth,
            )
            continue

        avg_depth = (depth_a + depth_b) / 2
        corrected_dist = pixel_dist * avg_depth / min_depth

        results.append(
            DepthCorrectedInterDistance(
                animal_a_id=a.animal_id,
                animal_a_category=a.category,
                animal_b_id=b.animal_id,
                animal_b_category=b.category,
                eye_a=eye_a,
                eye_b=eye_b,
                pixel_distance=round(pixel_dist, 2),
                depth_corrected_distance=round(corrected_dist, 2),
                depth_a=round(depth_a, 4),
                depth_b=round(depth_b, 4),
            )
        )

    return results


def measure_all(
    animals: list[AnimalDetection],
    depth_map: np.ndarray | None = None,
) -> tuple[
    list[IntraAnimalDistance],
    list[InterAnimalDistance],
    list[DepthCorrectedInterDistance] | None,
]:
    """Compute all distance measurements.

    Returns:
        (intra_distances, inter_distances, depth_corrected_inter_distances)
        depth_corrected is None if no depth_map provided.
    """
    intra = compute_intra_distances(animals)
    inter = compute_inter_distances(animals)

    depth_corrected = None
    if depth_map is not None:
        depth_corrected = compute_depth_corrected_inter_distances(animals, depth_map)

    return intra, inter, depth_corrected
