"""Extract minimal COCO annotations for test images.

Run this once (with the full COCO dataset available) to generate
a small annotation file that ships with the repo.

Usage:
    python -m scripts.extract_test_annotations
"""

import json
from pathlib import Path

from pycocotools.coco import COCO

# The 3 selected test images
TEST_IMAGE_IDS = [287545, 402473, 547383]

FULL_ANNOTATIONS = Path("data/coco/annotations/instances_val2017.json")
OUTPUT_PATH = Path("data/test_annotations/test_instances.json")


def main():
    print(f"Loading full COCO annotations from {FULL_ANNOTATIONS}")
    coco = COCO(str(FULL_ANNOTATIONS))

    # Get image info for our test images
    images = coco.loadImgs(TEST_IMAGE_IDS)

    # Get all annotations for these images (all categories, not just animals)
    ann_ids = coco.getAnnIds(imgIds=TEST_IMAGE_IDS)
    annotations = coco.loadAnns(ann_ids)

    # Get all categories referenced by these annotations
    cat_ids = list({a["category_id"] for a in annotations})
    categories = coco.loadCats(cat_ids)

    # Build minimal COCO-format JSON
    mini_coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(mini_coco, f, indent=2)

    print(f"Extracted {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")
    print(f"Saved to {OUTPUT_PATH}")

    # Summary
    for img in images:
        img_anns = [a for a in annotations if a["image_id"] == img["id"]]
        cats = [coco.loadCats(a["category_id"])[0]["name"] for a in img_anns]
        print(f"  {img['file_name']}: {len(img_anns)} annotations ({', '.join(cats)})")


if __name__ == "__main__":
    main()
