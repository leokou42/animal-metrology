from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # COCO dataset paths
    # Default to test data (ships with repo, no download needed)
    # Override with env vars to use the full COCO dataset
    coco_images_dir: Path = Path("./data/test_images")
    coco_annotations_file: Path = Path(
        "./data/test_annotations/test_instances.json"
    )

    # Segmentation model
    segmentation_model: str = "yolo11n-seg"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45

    # Eye detection model (RTMPose + AP-10K)
    eye_model_path: str = "weights/rtmpose_ap10k.onnx"
    eye_confidence_threshold: float = 0.3

    # Depth estimation (Depth Pro primary, DA V2 fallback)
    depth_pro_checkpoint: str = "weights/depth_pro.pt"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_enabled: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # Output
    output_dir: Path = Path("./outputs")

    # Pre-selected test image IDs (verified: >=2 animals, both eyes visible)
    test_image_ids: list[int] = [287545, 402473, 547383]

    # COCO animal category IDs (supercategory == "animal")
    # bird=16, cat=17, dog=18, horse=19, sheep=20, cow=21,
    # elephant=22, bear=23, zebra=24, giraffe=25
    animal_category_ids: list[int] = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
