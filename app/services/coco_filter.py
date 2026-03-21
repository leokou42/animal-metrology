"""COCO dataset filtering service.

Filters COCO val2017 images to find those containing >= 2 animal instances.
Uses pycocotools for efficient annotation querying.
"""

import logging
from collections import Counter
from pathlib import Path

from pycocotools.coco import COCO

from app.config import settings

logger = logging.getLogger(__name__)

# COCO animal category names for readable output
ANIMAL_CATEGORIES = {
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
}


class COCOFilterService:
    """Filters COCO dataset for images containing multiple animals."""

    def __init__(
        self,
        annotations_file: Path | None = None,
        images_dir: Path | None = None,
    ):
        self.annotations_file = annotations_file or settings.coco_annotations_file
        self.images_dir = images_dir or settings.coco_images_dir
        self._coco: COCO | None = None

    @property
    def coco(self) -> COCO:
        if self._coco is None:
            logger.info("Loading COCO annotations from %s", self.annotations_file)
            self._coco = COCO(str(self.annotations_file))
        return self._coco

    def get_animal_category_ids(self) -> list[int]:
        """Return category IDs for all animal supercategories in COCO."""
        cats = self.coco.loadCats(self.coco.getCatIds())
        animal_ids = [c["id"] for c in cats if c["supercategory"] == "animal"]
        logger.info(
            "Found %d animal categories: %s",
            len(animal_ids),
            [c["name"] for c in cats if c["id"] in animal_ids],
        )
        return animal_ids

    def filter_multi_animal_images(
        self,
        min_animals: int = 2,
        max_animals: int | None = None,
        category_filter: list[str] | None = None,
        max_results: int | None = None,
    ) -> list[dict]:
        """Find images containing between min_animals and max_animals instances.

        Args:
            min_animals: Minimum number of animal instances (inclusive).
            max_animals: Maximum number of animal instances (inclusive).
                         None means no upper bound.
            category_filter: Only include images containing these animal types.
                             e.g. ["cat", "dog", "elephant"]
            max_results: Cap on number of results returned.

        Returns a list of dicts with image metadata and animal annotation info.
        """
        animal_cat_ids = self.get_animal_category_ids()

        # Get all annotation IDs for animal categories
        ann_ids = self.coco.getAnnIds(catIds=animal_cat_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Group annotations by image_id
        image_animals: dict[int, list[dict]] = {}
        for ann in anns:
            img_id = ann["image_id"]
            if img_id not in image_animals:
                image_animals[img_id] = []
            image_animals[img_id].append(ann)

        # Filter images by animal count range and optional category
        results = []
        for img_id, animal_anns in image_animals.items():
            n = len(animal_anns)
            if n < min_animals:
                continue
            if max_animals is not None and n > max_animals:
                continue

            cat_counts = Counter(
                ANIMAL_CATEGORIES.get(a["category_id"], f"id_{a['category_id']}")
                for a in animal_anns
            )

            # If category filter is set, skip images without matching categories
            if category_filter:
                if not any(c in cat_counts for c in category_filter):
                    continue

            img_info = self.coco.loadImgs(img_id)[0]
            image_path = self.images_dir / img_info["file_name"]
            results.append({
                "image_id": img_id,
                "file_name": img_info["file_name"],
                "file_path": str(image_path),
                "width": img_info["width"],
                "height": img_info["height"],
                "num_animals": n,
                "animal_categories": dict(cat_counts),
                "annotation_ids": [a["id"] for a in animal_anns],
                "file_exists": image_path.exists(),
            })

        # Sort by number of animals (ascending) — fewer animals = better for our use case
        results.sort(key=lambda x: x["num_animals"])

        if max_results:
            results = results[:max_results]

        logger.info(
            "Found %d images with %d~%s animals",
            len(results),
            min_animals,
            max_animals or "∞",
        )
        return results

    def get_image_annotations(self, image_id: int) -> dict:
        """Get full annotation details for a specific image."""
        animal_cat_ids = self.get_animal_category_ids()
        ann_ids = self.coco.getAnnIds(
            imgIds=image_id, catIds=animal_cat_ids, iscrowd=False
        )
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(image_id)[0]

        return {
            "image_info": img_info,
            "annotations": anns,
            "num_animals": len(anns),
        }


# Module-level singleton
_filter_service: COCOFilterService | None = None


def get_coco_filter_service() -> COCOFilterService:
    global _filter_service
    if _filter_service is None:
        _filter_service = COCOFilterService()
    return _filter_service
