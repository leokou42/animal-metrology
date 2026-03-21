"""Run the full analysis pipeline on test images and export results.

Usage:
    python -m scripts.run_demo

Outputs:
    - Annotated images in outputs/
    - CSV results in data/sample_results.csv
    - Summary table printed to console
"""

import csv
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.models.schemas import AnimalDetection, EyePair
from app.services.segmentation import SegmentationService
from app.services.eye_detection import EyeDetectionService
from app.services.measurement import measure_all


def main():
    print("=" * 60)
    print("Animal Eye Metrology — Demo Pipeline")
    print("=" * 60)

    # Initialize services
    print("\n[1/4] Loading models...")
    seg_svc = SegmentationService()
    eye_svc = EyeDetectionService()

    # Try loading depth service (optional)
    depth_svc = None
    try:
        from app.services.depth_estimation import get_depth_estimation_service
        depth_svc = get_depth_estimation_service()
        if depth_svc:
            print("  Depth Anything V2 loaded")
        else:
            print("  Depth estimation disabled or unavailable")
    except Exception as e:
        print(f"  Depth estimation unavailable: {e}")

    print("  Models loaded successfully")

    # Prepare output
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path("data/sample_results.csv")
    csv_rows = []

    test_images = []
    for img_id in settings.test_image_ids:
        path = Path(settings.coco_images_dir) / f"{img_id:012d}.jpg"
        if path.exists():
            test_images.append((img_id, path))
        else:
            print(f"  WARNING: Image {img_id} not found at {path}, skipping")

    if not test_images:
        print("ERROR: No test images found. Check data/test_images/")
        sys.exit(1)

    print(f"\n[2/4] Processing {len(test_images)} images...")

    for img_id, img_path in test_images:
        print(f"\n--- Image {img_id} ---")

        # Segmentation
        seg_result = seg_svc.segment_animals(str(img_path))
        print(f"  Segmented {len(seg_result.animals)} animals")

        # Eye detection
        eye_pairs = eye_svc.detect_eyes(seg_result)

        # Build AnimalDetection list
        animals = [
            AnimalDetection(
                animal_id=a.animal_id,
                category=a.category,
                confidence=a.confidence,
                bbox=a.bbox,
                mask_area=a.mask_area,
                eyes=ep,
            )
            for a, ep in zip(seg_result.animals, eye_pairs)
        ]

        # Depth estimation (optional)
        depth_map = None
        if depth_svc and seg_result.raw_image is not None:
            try:
                depth_map = depth_svc.estimate_depth(seg_result.raw_image)
            except Exception as e:
                print(f"  Depth estimation failed: {e}")

        # Measurement
        intra, inter, depth_corrected = measure_all(animals, depth_map=depth_map)

        # Print results
        for a, ep in zip(seg_result.animals, eye_pairs):
            le = f"({ep.left_eye.x}, {ep.left_eye.y})" if ep.left_eye else "N/A"
            re = f"({ep.right_eye.x}, {ep.right_eye.y})" if ep.right_eye else "N/A"
            print(f"  #{a.animal_id} {a.category} (conf={a.confidence:.2f}): L={le}, R={re}")

        for d in intra:
            print(f"  Binocular #{d.animal_id}: {d.distance_px}px")

        for d in inter:
            print(f"  Inter #{d.animal_a_id}-#{d.animal_b_id} R-eye: {d.distance_px}px")

        if depth_corrected:
            for d in depth_corrected:
                print(f"  Depth-corrected #{d.animal_a_id}-#{d.animal_b_id}: {d.depth_corrected_distance}px")

        # Visualization
        from app.utils.visualization import visualize_results

        eye_data = [
            {
                "animal_id": a.animal_id,
                "left_eye": (ep.left_eye.x, ep.left_eye.y) if ep.left_eye else None,
                "right_eye": (ep.right_eye.x, ep.right_eye.y) if ep.right_eye else None,
            }
            for a, ep in zip(seg_result.animals, eye_pairs)
        ]

        # Build depth lookup for inter distance labels
        depth_lookup = {}
        if depth_corrected:
            for dc in depth_corrected:
                depth_lookup[(dc.animal_a_id, dc.animal_b_id)] = dc.depth_corrected_distance

        intra_dicts = [
            {
                "animal_id": d.animal_id,
                "left_eye": (d.left_eye.x, d.left_eye.y),
                "right_eye": (d.right_eye.x, d.right_eye.y),
                "distance_px": d.distance_px,
            }
            for d in intra
        ]
        inter_dicts = [
            {
                "animal_a_id": d.animal_a_id,
                "animal_b_id": d.animal_b_id,
                "eye_a": (d.eye_a.x, d.eye_a.y),
                "eye_b": (d.eye_b.x, d.eye_b.y),
                "distance_px": d.distance_px,
                "depth_corrected_distance": depth_lookup.get(
                    (d.animal_a_id, d.animal_b_id)
                ),
            }
            for d in inter
        ]

        output_path = settings.output_dir / f"analyze_{img_id}.jpg"
        visualize_results(
            seg_result=seg_result,
            eye_data=eye_data,
            intra_distances=intra_dicts,
            inter_distances=inter_dicts,
            output_path=output_path,
        )
        print(f"  Saved: {output_path}")

        # Build CSV rows
        img_file = f"{img_id:012d}.jpg"
        # Build inter-distance lookup by animal_id
        inter_by_pair = {(d.animal_a_id, d.animal_b_id): d for d in inter}
        depth_by_pair = {(d.animal_a_id, d.animal_b_id): d for d in (depth_corrected or [])}
        intra_by_id = {d.animal_id: d for d in intra}

        for a, ep in zip(seg_result.animals, eye_pairs):
            intra_d = intra_by_id.get(a.animal_id)
            row = {
                "image_id": img_id,
                "image_file": img_file,
                "animal_id": a.animal_id,
                "category": a.category,
                "confidence": round(a.confidence, 4),
                "bbox_x1": round(a.bbox[0], 1),
                "bbox_y1": round(a.bbox[1], 1),
                "bbox_x2": round(a.bbox[2], 1),
                "bbox_y2": round(a.bbox[3], 1),
                "left_eye_x": ep.left_eye.x if ep.left_eye else "",
                "left_eye_y": ep.left_eye.y if ep.left_eye else "",
                "right_eye_x": ep.right_eye.x if ep.right_eye else "",
                "right_eye_y": ep.right_eye.y if ep.right_eye else "",
                "binocular_distance_px": intra_d.distance_px if intra_d else "",
            }
            csv_rows.append(row)

        # Add inter-distance rows
        for d in inter:
            pair_key = (d.animal_a_id, d.animal_b_id)
            dc = depth_by_pair.get(pair_key)
            csv_rows.append({
                "image_id": img_id,
                "image_file": img_file,
                "animal_id": "",
                "category": "",
                "confidence": "",
                "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
                "left_eye_x": "", "left_eye_y": "",
                "right_eye_x": "", "right_eye_y": "",
                "binocular_distance_px": "",
                "inter_animal_pair": f"#{d.animal_a_id}({d.animal_a_category})-#{d.animal_b_id}({d.animal_b_category})",
                "inter_animal_right_eye_distance_px": d.distance_px,
                "depth_corrected_inter_distance": dc.depth_corrected_distance if dc else "",
            })

    # Write CSV
    print(f"\n[3/4] Writing CSV to {csv_path}...")
    fieldnames = [
        "image_id", "image_file", "animal_id", "category", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
        "binocular_distance_px",
        "inter_animal_pair", "inter_animal_right_eye_distance_px",
        "depth_corrected_inter_distance",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"  {len(csv_rows)} rows written")

    print("\n[4/4] Done!")
    print(f"  Annotated images: {settings.output_dir}/")
    print(f"  CSV results: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
