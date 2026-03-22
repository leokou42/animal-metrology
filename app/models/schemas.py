from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class EyePair(BaseModel):
    left_eye: Point | None = None
    right_eye: Point | None = None


class AnimalDetection(BaseModel):
    animal_id: int
    category: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]
    mask_area: int
    eyes: EyePair


class IntraAnimalDistance(BaseModel):
    """Binocular distance for a single animal (left_eye → right_eye)."""
    animal_id: int
    category: str
    left_eye: Point
    right_eye: Point
    # Layer 1: pixel distance
    pixel_distance: float
    # Layer 2a: depth-corrected pixel distance (relative depth, DA V2)
    depth_corrected_px: float | None = None
    # Layer 2b: metric distance (Depth Pro only)
    metric_distance_m: float | None = None
    depth_left_eye_m: float | None = None
    depth_right_eye_m: float | None = None
    focal_length_px: float | None = None
    # Layer 3: sanity check against known IOD
    known_iod_range_cm: list[float] | None = None
    sanity_check_result: str | None = None  # PASS / WARNING / FAIL


class InterAnimalDistance(BaseModel):
    """Distance between right eyes of two animals."""
    animal_a_id: int
    animal_a_category: str
    animal_b_id: int
    animal_b_category: str
    eye_a: Point
    eye_b: Point
    # Layer 1: pixel distance
    pixel_distance: float
    # Layer 2a: depth-corrected pixel distance (relative depth, DA V2)
    depth_corrected_px: float | None = None
    # Layer 2b: metric distance (Depth Pro only)
    metric_distance_m: float | None = None


class MeasurementResult(BaseModel):
    image_id: int | str
    image_file: str
    image_width: int
    image_height: int
    animals: list[AnimalDetection]
    intra_distances: list[IntraAnimalDistance]
    inter_distances: list[InterAnimalDistance]
    annotated_image_path: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class COCOFilterResult(BaseModel):
    total_images_found: int
    sample_images: list[dict]
