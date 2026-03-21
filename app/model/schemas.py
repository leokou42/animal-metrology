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
    animal_id: int
    category: str
    left_eye: Point
    right_eye: Point
    distance_px: float


class InterAnimalDistance(BaseModel):
    animal_a_id: int
    animal_a_category: str
    animal_b_id: int
    animal_b_category: str
    eye_a: Point
    eye_b: Point
    distance_px: float


class DepthCorrectedInterDistance(BaseModel):
    """Depth-corrected distance between two animals' right eyes.

    Each entry represents one pair from itertools.combinations.
    e.g., 3 animals → 3 pairs: (0,1), (0,2), (1,2)
    """
    animal_a_id: int
    animal_a_category: str
    animal_b_id: int
    animal_b_category: str
    eye_a: Point
    eye_b: Point
    pixel_distance: float
    depth_corrected_distance: float
    depth_a: float
    depth_b: float


class MeasurementResult(BaseModel):
    image_id: int
    image_file: str
    image_width: int
    image_height: int
    animals: list[AnimalDetection]
    intra_distances: list[IntraAnimalDistance]
    inter_distances: list[InterAnimalDistance]
    depth_corrected_inter_distances: list[DepthCorrectedInterDistance] | None = None
    annotated_image_path: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class COCOFilterResult(BaseModel):
    total_images_found: int
    sample_images: list[dict]
