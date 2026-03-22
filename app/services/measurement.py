"""Three-layer distance measurement service for animal eye metrology.

=== Three Layers ===
Layer 1 (2D Pixel): Euclidean distance in image pixel space
Layer 2 (Metric):   3D distance in meters using depth map + focal length
Layer 3 (Sanity):   Cross-check metric IOD against known biological ranges

=== Data Flow ===
AnimalDetection list + optional DepthResult → distance calculations → results

=== Units ===
Layer 1: pixels (no physical meaning, affected by perspective)
Layer 2: meters (requires metric depth model like Depth Pro)
Layer 3: PASS / WARNING / FAIL (validation only)
"""

import logging
import math
from itertools import combinations

import numpy as np

from app.models.schemas import (
    AnimalDetection,
    InterAnimalDistance,
    IntraAnimalDistance,
    Point,
)

logger = logging.getLogger(__name__)

# ============================================================
# Known Inter-Ocular Distance (IOD) for sanity checking
# Source: veterinary anatomy references, approximate adult values
# ============================================================
KNOWN_IOD_CM = {
    "cat": (5.0, 6.0),
    "dog": (6.0, 10.0),
    "giraffe": (18.0, 22.0),
    "horse": (16.0, 20.0),
    "cow": (15.0, 20.0),
    "elephant": (22.0, 28.0),
    "sheep": (6.0, 8.0),
    "zebra": (14.0, 18.0),
    "bear": (12.0, 16.0),
    "bird": (1.0, 4.0),
}

# Allow 50% tolerance for metric depth estimation inaccuracy
SANITY_TOLERANCE = 0.5


def compute_euclidean_distance(p1: Point, p2: Point) -> float:
    """2D Euclidean distance between two points in pixel space."""
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def compute_metric_distance(
    p1: Point,
    p2: Point,
    depth_map: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    """Compute 3D metric distance between two points using depth + focal length.

    Projects pixel coordinates to 3D camera coordinates:
        X = (x_pixel - cx) * Z / focal_length
        Y = (y_pixel - cy) * Z / focal_length
        Z = depth_map[y][x]

    Args:
        p1, p2: pixel coordinates
        depth_map: metric depth map (H, W) in meters
        focal_length: estimated focal length in pixels
        cx, cy: principal point (image center)

    Returns:
        (distance_m, depth1, depth2) — 3D distance in meters and depth values
    """
    h, w = depth_map.shape[:2]

    # Clamp to image bounds
    x1, y1 = int(min(max(p1.x, 0), w - 1)), int(min(max(p1.y, 0), h - 1))
    x2, y2 = int(min(max(p2.x, 0), w - 1)), int(min(max(p2.y, 0), h - 1))

    z1 = float(depth_map[y1, x1])
    z2 = float(depth_map[y2, x2])

    # Project to 3D
    X1 = (p1.x - cx) * z1 / focal_length
    Y1 = (p1.y - cy) * z1 / focal_length
    X2 = (p2.x - cx) * z2 / focal_length
    Y2 = (p2.y - cy) * z2 / focal_length

    dist = math.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (z2 - z1) ** 2)
    return dist, z1, z2


def compute_depth_corrected_px(
    p1: Point,
    p2: Point,
    pixel_distance: float,
    depth_map: np.ndarray,
) -> float:
    """Correct pixel distance using relative depth to compensate for perspective.

    When two points are at different depths, the closer one appears larger in
    the image. This function normalizes the pixel distance by the depth ratio
    so that objects at different distances are more fairly compared.

    Formula: corrected = pixel_dist * avg_depth / min(depth_a, depth_b)
    """
    h, w = depth_map.shape[:2]
    x1, y1 = int(min(max(p1.x, 0), w - 1)), int(min(max(p1.y, 0), h - 1))
    x2, y2 = int(min(max(p2.x, 0), w - 1)), int(min(max(p2.y, 0), h - 1))

    d1 = float(depth_map[y1, x1])
    d2 = float(depth_map[y2, x2])

    min_depth = min(d1, d2)
    if min_depth < 1e-6:
        return pixel_distance

    avg_depth = (d1 + d2) / 2.0
    return pixel_distance * avg_depth / min_depth


def sanity_check(category: str, metric_distance_m: float) -> str | None:
    """Check if measured IOD falls within expected biological range.

    Returns:
        "PASS" — within range
        "WARNING" — outside range but within tolerance
        "FAIL" — clearly unreasonable
        None — no reference data for this category
    """
    if category not in KNOWN_IOD_CM:
        return None

    low_cm, high_cm = KNOWN_IOD_CM[category]
    measured_cm = metric_distance_m * 100

    if low_cm <= measured_cm <= high_cm:
        return "PASS"

    low_tol = low_cm * (1 - SANITY_TOLERANCE)
    high_tol = high_cm * (1 + SANITY_TOLERANCE)

    if low_tol <= measured_cm <= high_tol:
        return "WARNING"

    return "FAIL"


def compute_intra_distances(
    animals: list[AnimalDetection],
    depth_map: np.ndarray | None = None,
    focal_length: float | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[IntraAnimalDistance]:
    """Compute binocular distance for each animal (all 3 layers).

    Animals missing either eye are skipped.
    """
    results: list[IntraAnimalDistance] = []
    has_metric = depth_map is not None and focal_length is not None

    cx, cy = 0.0, 0.0
    if has_metric and image_size:
        cx = image_size[1] / 2.0  # width / 2
        cy = image_size[0] / 2.0  # height / 2

    for animal in animals:
        if animal.eyes.left_eye is None or animal.eyes.right_eye is None:
            continue

        pixel_dist = compute_euclidean_distance(
            animal.eyes.left_eye, animal.eyes.right_eye
        )

        corrected_px = None
        metric_dist = None
        depth_l = None
        depth_r = None
        check_result = None
        iod_range = None

        # Depth-corrected pixel distance (works with any depth map)
        if depth_map is not None:
            corrected_px = compute_depth_corrected_px(
                animal.eyes.left_eye, animal.eyes.right_eye,
                pixel_dist, depth_map,
            )

        # Metric 3D distance (requires focal length from Depth Pro)
        if has_metric:
            metric_dist, depth_l, depth_r = compute_metric_distance(
                animal.eyes.left_eye, animal.eyes.right_eye,
                depth_map, focal_length, cx, cy,
            )
            check_result = sanity_check(animal.category, metric_dist)
            if animal.category in KNOWN_IOD_CM:
                iod_range = list(KNOWN_IOD_CM[animal.category])

        results.append(
            IntraAnimalDistance(
                animal_id=animal.animal_id,
                category=animal.category,
                left_eye=animal.eyes.left_eye,
                right_eye=animal.eyes.right_eye,
                pixel_distance=round(pixel_dist, 2),
                depth_corrected_px=round(corrected_px, 2) if corrected_px is not None else None,
                metric_distance_m=round(metric_dist, 4) if metric_dist is not None else None,
                depth_left_eye_m=round(depth_l, 4) if depth_l is not None else None,
                depth_right_eye_m=round(depth_r, 4) if depth_r is not None else None,
                focal_length_px=round(focal_length, 1) if focal_length is not None else None,
                known_iod_range_cm=iod_range,
                sanity_check_result=check_result,
            )
        )

    return results


def compute_inter_distances(
    animals: list[AnimalDetection],
    depth_map: np.ndarray | None = None,
    focal_length: float | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[InterAnimalDistance]:
    """Compute pairwise right-eye distances between all animals (Layer 1 + 2)."""
    eligible = [a for a in animals if a.eyes.right_eye is not None]
    has_metric = depth_map is not None and focal_length is not None

    cx, cy = 0.0, 0.0
    if has_metric and image_size:
        cx = image_size[1] / 2.0
        cy = image_size[0] / 2.0

    results: list[InterAnimalDistance] = []

    for a, b in combinations(eligible, 2):
        pixel_dist = compute_euclidean_distance(a.eyes.right_eye, b.eyes.right_eye)

        corrected_px = None
        metric_dist = None

        if depth_map is not None:
            corrected_px = compute_depth_corrected_px(
                a.eyes.right_eye, b.eyes.right_eye,
                pixel_dist, depth_map,
            )

        if has_metric:
            metric_dist, _, _ = compute_metric_distance(
                a.eyes.right_eye, b.eyes.right_eye,
                depth_map, focal_length, cx, cy,
            )

        results.append(
            InterAnimalDistance(
                animal_a_id=a.animal_id,
                animal_a_category=a.category,
                animal_b_id=b.animal_id,
                animal_b_category=b.category,
                eye_a=a.eyes.right_eye,
                eye_b=b.eyes.right_eye,
                pixel_distance=round(pixel_dist, 2),
                depth_corrected_px=round(corrected_px, 2) if corrected_px is not None else None,
                metric_distance_m=round(metric_dist, 4) if metric_dist is not None else None,
            )
        )

    return results


def measure_all(
    animals: list[AnimalDetection],
    depth_map: np.ndarray | None = None,
    focal_length: float | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[IntraAnimalDistance], list[InterAnimalDistance]]:
    """Compute all distance measurements (3 layers)."""
    intra = compute_intra_distances(animals, depth_map, focal_length, image_size)
    inter = compute_inter_distances(animals, depth_map, focal_length, image_size)
    return intra, inter
