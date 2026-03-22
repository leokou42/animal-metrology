"""Run the full analysis pipeline on test images and export results.

Usage:
    python -m scripts.run_demo

Outputs:
    - Annotated images: outputs/analyze_{image_id}.jpg
    - CSV: data/sample_results.csv
    - Console summary table
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.models.schemas import AnimalDetection
from app.services.segmentation import SegmentationService
from app.services.eye_detection import EyeDetectionService
from app.services.measurement import measure_all



def _print_table(image_results):
    """Print a formatted results table to console."""
    W = 72

    print(f"\n{'=' * W}")
    print(f"{'Animal Eye Metrology Results':^{W}}")
    print(f"{'=' * W}")

    for img_id, intra, inter in image_results:
        if not intra and not inter:
            print(f"\n  Image {img_id}: No animals detected")
            continue

        categories = ", ".join(
            f"{d.category}" for d in intra
        ) if intra else "unknown"
        print(f"\n  Image: {img_id} ({categories})")
        print(f"  {'─' * (W - 4)}")
        print(f"  {'Animal':<10} {'Category':<10} {'Pixel Dist':<12} {'Metric Est':<12} {'Sanity Check'}")
        print(f"  {'─' * (W - 4)}")

        for d in intra:
            metric = f"{d.metric_distance_m:.2f} m" if d.metric_distance_m else "N/A"
            check = d.sanity_check_result or "N/A"
            iod = f"IOD {d.known_iod_range_cm[0]:.0f}-{d.known_iod_range_cm[1]:.0f}cm" if d.known_iod_range_cm else ""
            icon = {"PASS": "v", "WARNING": "!", "FAIL": "x"}.get(check, " ")
            print(f"  #{d.animal_id:<9} {d.category:<10} {d.pixel_distance:<12.1f} {metric:<12} {icon} {iod}")

        if inter:
            print(f"  {'─' * (W - 4)}")
            print(f"  Inter-animal (right eye)")
            print(f"  {'Pair':<10} {'Animals':<18} {'Pixel Dist':<12} {'Metric Est'}")
            print(f"  {'─' * (W - 4)}")

            for d in inter:
                pair = f"#{d.animal_a_id}<->{d.animal_b_id}"
                animals = f"{d.animal_a_category}-{d.animal_b_category}"
                metric = f"{d.metric_distance_m:.2f} m" if d.metric_distance_m else "N/A"
                print(f"  {pair:<10} {animals:<18} {d.pixel_distance:<12.1f} {metric}")

    print(f"\n{'=' * W}")


def main():
    print("=" * 60)
    print("Animal Eye Metrology — Demo Pipeline")
    print("=" * 60)

    # Initialize services
    print("\n[1/4] Loading models...")
    seg_svc = SegmentationService()
    eye_svc = EyeDetectionService()

    depth_svc = None
    try:
        from app.services.depth_estimation import get_depth_estimation_service
        depth_svc = get_depth_estimation_service()
        if depth_svc:
            print(f"  Depth model: {depth_svc.backend_name}")
        else:
            print("  Depth estimation disabled or unavailable")
    except Exception as e:
        print(f"  Depth estimation unavailable: {e}")

    print("  Models loaded successfully")

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path("data/sample_results.csv")
    csv_rows = []
    image_results = []

    test_images = []
    for img_id in settings.test_image_ids:
        path = Path(settings.coco_images_dir) / f"{img_id:012d}.jpg"
        if path.exists():
            test_images.append((img_id, path))
        else:
            print(f"  WARNING: Image {img_id} not found at {path}")

    if not test_images:
        print("ERROR: No test images found")
        sys.exit(1)

    print(f"\n[2/4] Processing {len(test_images)} images...")

    for img_id, img_path in test_images:
        print(f"\n  Processing {img_id}...", end="", flush=True)

        seg_result = seg_svc.segment_animals(str(img_path))
        eye_pairs = eye_svc.detect_eyes(seg_result)

        # Depth estimation
        depth_map = None
        focal_length = None
        depth_result = None
        if depth_svc and seg_result.raw_image is not None:
            try:
                depth_result = depth_svc.estimate_depth(seg_result.raw_image)
                depth_map = depth_result.depth_map
                if depth_result.is_metric:
                    focal_length = depth_result.focal_length_px

            except Exception as e:
                print(f" depth failed: {e}", end="")

        animals = [
            AnimalDetection(
                animal_id=a.animal_id, category=a.category,
                confidence=a.confidence, bbox=a.bbox,
                mask_area=a.mask_area, eyes=ep,
            )
            for a, ep in zip(seg_result.animals, eye_pairs)
        ]

        intra, inter = measure_all(
            animals, depth_map=depth_map,
            focal_length=focal_length, image_size=seg_result.image_hw,
        )

        image_results.append((img_id, intra, inter))

        # Visualization
        from app.utils.visualization import visualize_results

        eye_data = [
            {"animal_id": a.animal_id,
             "left_eye": (ep.left_eye.x, ep.left_eye.y) if ep.left_eye else None,
             "right_eye": (ep.right_eye.x, ep.right_eye.y) if ep.right_eye else None}
            for a, ep in zip(seg_result.animals, eye_pairs)
        ]
        intra_dicts = [
            {"animal_id": d.animal_id, "category": d.category,
             "left_eye": (d.left_eye.x, d.left_eye.y),
             "right_eye": (d.right_eye.x, d.right_eye.y),
             "distance_px": d.pixel_distance,
             "depth_corrected_px": d.depth_corrected_px,
             "metric_distance_m": d.metric_distance_m,
             "sanity_check_result": d.sanity_check_result}
            for d in intra
        ]
        inter_dicts = [
            {"animal_a_id": d.animal_a_id, "animal_b_id": d.animal_b_id,
             "eye_a": (d.eye_a.x, d.eye_a.y), "eye_b": (d.eye_b.x, d.eye_b.y),
             "distance_px": d.pixel_distance,
             "depth_corrected_px": d.depth_corrected_px,
             "metric_distance_m": d.metric_distance_m}
            for d in inter
        ]

        output_path = settings.output_dir / f"analyze_{img_id}.jpg"
        visualize_results(
            seg_result=seg_result, eye_data=eye_data,
            intra_distances=intra_dicts, inter_distances=inter_dicts,
            output_path=output_path,
        )

        print(f" done ({len(seg_result.animals)} animals)")

        # CSV rows
        img_file = f"{img_id:012d}.jpg"
        intra_by_id = {d.animal_id: d for d in intra}

        for a, ep in zip(seg_result.animals, eye_pairs):
            d = intra_by_id.get(a.animal_id)
            csv_rows.append({
                "image_id": img_id, "image_file": img_file,
                "animal_id": a.animal_id, "category": a.category,
                "confidence": round(a.confidence, 4),
                "bbox_x1": round(a.bbox[0], 1), "bbox_y1": round(a.bbox[1], 1),
                "bbox_x2": round(a.bbox[2], 1), "bbox_y2": round(a.bbox[3], 1),
                "left_eye_x": ep.left_eye.x if ep.left_eye else "",
                "left_eye_y": ep.left_eye.y if ep.left_eye else "",
                "right_eye_x": ep.right_eye.x if ep.right_eye else "",
                "right_eye_y": ep.right_eye.y if ep.right_eye else "",
                "binocular_distance_px": d.pixel_distance if d else "",
                "binocular_distance_metric_m": d.metric_distance_m if d and d.metric_distance_m else "",
                "depth_left_eye_m": d.depth_left_eye_m if d and d.depth_left_eye_m else "",
                "depth_right_eye_m": d.depth_right_eye_m if d and d.depth_right_eye_m else "",
                "focal_length_px": d.focal_length_px if d and d.focal_length_px else "",
                "known_iod_range_cm": f"{d.known_iod_range_cm[0]}-{d.known_iod_range_cm[1]}" if d and d.known_iod_range_cm else "",
                "sanity_check_result": d.sanity_check_result if d else "",
            })

        for d in inter:
            csv_rows.append({
                "image_id": img_id, "image_file": img_file,
                "inter_animal_pair": f"#{d.animal_a_id}({d.animal_a_category})-#{d.animal_b_id}({d.animal_b_category})",
                "inter_animal_distance_px": d.pixel_distance,
                "inter_animal_distance_metric_m": d.metric_distance_m if d.metric_distance_m else "",
            })

    # Write CSV
    print(f"\n[3/4] Writing CSV to {csv_path}...")
    fieldnames = [
        "image_id", "image_file", "animal_id", "category", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
        "binocular_distance_px", "binocular_distance_metric_m",
        "depth_left_eye_m", "depth_right_eye_m", "focal_length_px",
        "known_iod_range_cm", "sanity_check_result",
        "inter_animal_pair", "inter_animal_distance_px", "inter_animal_distance_metric_m",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"  {len(csv_rows)} rows written")

    # Print results table
    print("\n[4/4] Results summary:")
    _print_table(image_results)

    print(f"\n  Annotated images: {settings.output_dir}/")
    print(f"  CSV results: {csv_path}")


if __name__ == "__main__":
    main()
