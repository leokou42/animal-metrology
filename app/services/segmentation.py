"""Instance segmentation service using YOLOv11-seg.

=== 這個 module 的職責 ===
接收一張圖片路徑，跑 YOLO instance segmentation 模型，
回傳每隻動物的：
  - bounding box (x1, y1, x2, y2)
  - segmentation mask (binary numpy array, 跟原圖同尺寸)
  - category name (cat, dog, elephant...)
  - confidence score

=== 為什麼選 YOLO11-seg ===
1. COCO pretrained — 不需要額外 fine-tune，直接能辨識 80 類物體（含 10 種動物）
2. 輕量快速 — nano 版本 (yolo11n-seg) 在 CPU 上也能跑，面試官不需要 GPU
3. 一步到位 — 同時輸出 bbox + mask，不需要分兩個模型
4. ultralytics API 簡潔 — model.predict() 一行搞定

=== 資料流 ===
圖片路徑 → YOLO predict → 過濾出動物類別 → 過濾 confidence → 回傳結構化結果
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.config import settings

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ============================================================
# COCO class ID → name 對照表 (只列動物類別)
# ============================================================
# YOLO 用的 class index 是 0-based (跟 COCO 的 category_id 不同)
# 例如 COCO category_id=17 是 cat，但 YOLO 的 class index=15
# 這個 mapping 是 ultralytics 內建的，對應 COCO 80 classes
YOLO_ANIMAL_CLASS_IDS = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


@dataclass
class SegmentedAnimal:
    """一隻被偵測到的動物的完整資訊。

    Attributes:
        animal_id:  在這張圖裡的流水編號 (0, 1, 2, ...)
        category:   動物類別名稱 (cat, dog, ...)
        class_id:   YOLO 的 class index
        confidence: 模型對這個偵測的信心分數 (0~1)
        bbox:       bounding box [x1, y1, x2, y2]，pixel 座標
        mask:       binary mask，shape=(H, W)，1=動物區域, 0=背景
                    跟原圖尺寸完全一致，可以直接疊在原圖上
        mask_area:  mask 中為 1 的 pixel 數量，代表動物佔的面積
    """
    animal_id: int
    category: str
    class_id: int
    confidence: float
    bbox: list[float]
    mask: np.ndarray
    mask_area: int = 0

    def __post_init__(self):
        # mask_area 如果沒給，就從 mask 算
        if self.mask_area == 0 and self.mask is not None:
            self.mask_area = int(np.sum(self.mask > 0))


@dataclass
class SegmentationResult:
    """一張圖片的 segmentation 完整結果。

    Attributes:
        image_path: 原圖路徑
        image_hw:   (height, width) tuple
        animals:    偵測到的動物列表
        raw_image:  原圖的 numpy array (BGR format, OpenCV 預設)
    """
    image_path: str
    image_hw: tuple[int, int]
    animals: list[SegmentedAnimal] = field(default_factory=list)
    raw_image: np.ndarray | None = None


class SegmentationService:
    """YOLO-based instance segmentation for animals.

    === 生命週期 ===
    1. __init__: 載入 YOLO 模型 (第一次會自動下載權重)
    2. segment_animals(): 對一張圖跑推論，回傳 SegmentationResult
    3. 模型在整個 API 生命週期中只載入一次 (singleton pattern)

    === 關鍵設計決策 ===
    - 只回傳動物類別的偵測結果，忽略 person、car 等非動物物體
    - confidence threshold 預設 0.5，太低會有 false positive
    - mask 會被 resize 回原圖尺寸 (YOLO 內部用 640x640 跑推論)
    """

    def __init__(
        self,
        model_name: str | None = None,
        confidence: float | None = None,
        iou_threshold: float | None = None,
    ):
        """載入 YOLO segmentation 模型。

        Args:
            model_name: YOLO 模型名稱，例如 "yolo11n-seg"
                        n=nano(最快), s=small, m=medium, l=large, x=最精準
            confidence: 過濾門檻，低於此分數的偵測會被丟棄
            iou_threshold: NMS (Non-Maximum Suppression) 的 IoU 門檻
                           用來合併重疊的 bounding box
        """
        self.model_name = model_name or settings.segmentation_model
        self.confidence = confidence if confidence is not None else settings.confidence_threshold
        self.iou_threshold = iou_threshold if iou_threshold is not None else settings.iou_threshold

        logger.info("Loading YOLO model: %s", self.model_name)
        # Lazy import — only loaded when service is actually instantiated
        from ultralytics import YOLO

        # ultralytics 會自動下載權重到 ~/.cache/ultralytics/
        # 第一次跑會花幾秒下載，之後就會用 cache
        self.model = YOLO(self.model_name)
        logger.info("YOLO model loaded successfully")

    def segment_animals(self, image_path: str | Path) -> SegmentationResult:
        """對一張圖片執行 instance segmentation，只回傳動物。

        === 內部流程 ===
        1. cv2.imread 讀圖 → numpy array (BGR)
        2. model.predict() 跑 YOLO 推論
           - 內部會 resize 到 640x640，跑完再把座標映射回原圖尺寸
           - 回傳 Results 物件，包含 boxes + masks
        3. 遍歷每個偵測結果：
           a. 檢查 class_id 是否在 YOLO_ANIMAL_CLASS_IDS 裡
           b. 檢查 confidence 是否 >= threshold
           c. 從 Results.masks.data 取出 binary mask
           d. 把 mask resize 回原圖尺寸 (YOLO 內部用較小尺寸)
           e. 包裝成 SegmentedAnimal dataclass
        4. 按 confidence 降序排列，回傳 SegmentationResult

        Args:
            image_path: 圖片檔案路徑

        Returns:
            SegmentationResult 包含所有偵測到的動物

        Raises:
            FileNotFoundError: 圖片不存在
            ValueError: 圖片讀取失敗 (格式錯誤等)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # === Step 1: 讀取圖片 ===
        # cv2.imread 回傳 BGR numpy array, shape=(H, W, 3)
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        h, w = image.shape[:2]
        logger.info("Processing image: %s (%dx%d)", image_path.name, w, h)

        # === Step 2: 跑 YOLO 推論 ===
        # verbose=False 關掉 YOLO 的 console 輸出
        # retina_masks=True 讓 mask 以原圖解析度輸出，不是模型內部的低解析度
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
            retina_masks=True,
        )

        # results 是一個 list，每張圖一個 Results 物件
        # 我們只傳了一張圖，所以取 results[0]
        result = results[0]

        animals: list[SegmentedAnimal] = []
        animal_id_counter = 0

        # === Step 3: 遍歷偵測結果，過濾出動物 ===
        # result.boxes: 每個偵測的 bounding box 資訊
        # result.masks: 每個偵測的 segmentation mask
        # 兩者的 index 是對應的: boxes[i] 和 masks[i] 是同一個物體

        if result.boxes is None or result.masks is None:
            logger.warning("No detections or masks found in image")
            return SegmentationResult(
                image_path=str(image_path),
                image_hw=(h, w),
                animals=[],
                raw_image=image,
            )

        for i in range(len(result.boxes)):
            # --- 3a: 取出 class_id 和 confidence ---
            class_id = int(result.boxes.cls[i].item())
            conf = float(result.boxes.conf[i].item())

            # --- 3b: 只保留動物類別 ---
            if class_id not in YOLO_ANIMAL_CLASS_IDS:
                continue

            category = YOLO_ANIMAL_CLASS_IDS[class_id]

            # --- 3c: 取出 bounding box ---
            # boxes.xyxy 格式: [x1, y1, x2, y2]，已經是原圖座標
            bbox = result.boxes.xyxy[i].cpu().numpy().tolist()

            # --- 3d: 取出 segmentation mask ---
            # masks.data shape: (N, H_mask, W_mask)
            # 因為用了 retina_masks=True，H_mask 和 W_mask 應該等於原圖尺寸
            mask = result.masks.data[i].cpu().numpy()

            # 安全起見，如果尺寸不一致就 resize
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(
                    mask, (w, h), interpolation=cv2.INTER_NEAREST
                )

            # 轉成 binary: YOLO 輸出的 mask 值在 0~1 之間
            # 用 0.5 當 threshold 轉成 0/1
            binary_mask = (mask > 0.5).astype(np.uint8)

            # --- 3e: 包裝成 SegmentedAnimal ---
            animals.append(
                SegmentedAnimal(
                    animal_id=animal_id_counter,
                    category=category,
                    class_id=class_id,
                    confidence=round(conf, 4),
                    bbox=[round(v, 1) for v in bbox],
                    mask=binary_mask,
                )
            )
            animal_id_counter += 1

        # 按 confidence 降序排列
        animals.sort(key=lambda a: a.confidence, reverse=True)
        # 重新編號
        for idx, animal in enumerate(animals):
            animal.animal_id = idx

        logger.info(
            "Found %d animals: %s",
            len(animals),
            [(a.category, a.confidence) for a in animals],
        )

        return SegmentationResult(
            image_path=str(image_path),
            image_hw=(h, w),
            animals=animals,
            raw_image=image,
        )


# ============================================================
# Singleton pattern — 避免每次 API request 都重新載入模型
# ============================================================
# YOLO 模型載入需要幾秒（讀權重檔到記憶體），
# 所以用 singleton 確保整個 API 生命週期只載入一次。
_seg_service: SegmentationService | None = None


def get_segmentation_service() -> SegmentationService:
    """取得或建立 SegmentationService singleton。"""
    global _seg_service
    if _seg_service is None:
        _seg_service = SegmentationService()
    return _seg_service
