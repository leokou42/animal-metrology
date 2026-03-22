"""Visualization utilities for drawing segmentation and measurement results.

=== 這個 module 的職責 ===
把 pipeline 各階段的輸出「畫」在原圖上，產生一張可供人眼驗證的 annotated image。
分成三層繪製，每層獨立可選：
  1. Segmentation layer — mask 半透明疊色 + bounding box + category label
  2. Eye keypoint layer — 左右眼圓點 + "L"/"R" 標籤
  3. Distance layer    — 雙眼連線 + inter-animal 右眼虛線 + 距離數字

=== 設計原則 ===
- 每隻動物分配不同顏色，用 COLOR_PALETTE 循環
- mask 用 alpha blending（透明度 0.4）疊在原圖上，不遮住細節
- 所有文字都有黑色背景框，確保在任何底色上都看得清楚
- 函式接受 Optional 參數，這樣可以只畫已完成的部分（例如目前只有 segmentation）
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from app.services.segmentation import SegmentationResult

logger = logging.getLogger(__name__)

# ============================================================
# 顏色配置
# ============================================================
# BGR format (OpenCV 預設)
# 每隻動物分配一個顏色，超過就循環使用
COLOR_PALETTE = [
    (0, 200, 0),      # green
    (255, 100, 0),     # blue-ish
    (0, 100, 255),     # orange-red
    (255, 255, 0),     # cyan
    (200, 0, 200),     # magenta
    (0, 255, 255),     # yellow
    (255, 0, 100),     # purple-blue
    (100, 255, 100),   # light green
]

# Mask overlay transparency (0=invisible, 1=opaque)
MASK_ALPHA = 0.4

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.6      # category label on bbox
FONT_SCALE_EYE = 0.45       # "L" / "R" near eye points
FONT_SCALE_DISTANCE = 0.5   # distance value on lines
FONT_THICKNESS = 1
FONT_COLOR = (255, 255, 255)  # white text
BG_COLOR = (0, 0, 0)          # black background for text


def get_color(index: int) -> tuple[int, int, int]:
    """Get a color from the palette by index (wraps around)."""
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]


def draw_text_with_bg(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = FONT_SCALE_LABEL,
    color: tuple[int, int, int] = FONT_COLOR,
    thickness: int = FONT_THICKNESS,
) -> None:
    """Draw text with a filled background rectangle for readability.

    在文字後面畫一個黑色矩形，確保不管底色是什麼都看得清楚。
    origin 是文字左下角的座標。座標會自動 clamp 到圖片範圍內。
    """
    img_h, img_w = image.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = origin

    pad = 2
    # Clamp x: keep text + padding within image width
    x = max(pad, min(x, img_w - text_w - pad))
    # Clamp y: keep text + padding within image height
    # text_h + 4 is the space above origin; baseline + 2 is below
    y = max(text_h + pad + 2, min(y, img_h - baseline - pad))

    # Background rectangle (slightly padded)
    cv2.rectangle(
        image,
        (x - pad, y - text_h - 4),
        (x + text_w + pad, y + baseline + 2),
        BG_COLOR,
        cv2.FILLED,
    )
    cv2.putText(image, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw_segmentation(
    image: np.ndarray,
    seg_result: SegmentationResult,
) -> np.ndarray:
    """在原圖上繪製 segmentation masks + bounding boxes + category labels。

    === 繪製邏輯 ===
    對每隻動物：
    1. Mask overlay — 建立一個跟原圖同尺寸的色彩圖層，只在 mask=1 的區域填色，
       然後用 cv2.addWeighted 做 alpha blending 疊回原圖
    2. Mask contour — 用 cv2.findContours 找出 mask 的邊緣，畫成實線，
       這就是題目要求的「動物輪廓」
    3. Bounding box — 用 bbox 座標畫矩形框
    4. Label — 在 bbox 左上角標註 "#{id} {category} {confidence}"

    Args:
        image: 原圖 (BGR, will be modified in-place)
        seg_result: segmentation 結果

    Returns:
        標註後的圖片 (same reference as input, modified in-place)
    """
    for animal in seg_result.animals:
        color = get_color(animal.animal_id)

        # --- 1. Mask overlay (semi-transparent fill) ---
        # 建一個純色圖層
        overlay = image.copy()
        # 只在 mask=1 的位置填上顏色
        overlay[animal.mask == 1] = color
        # Alpha blend 回原圖
        cv2.addWeighted(overlay, MASK_ALPHA, image, 1 - MASK_ALPHA, 0, image)

        # --- 2. Mask contour (輪廓線) ---
        contours, _ = cv2.findContours(
            animal.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, thickness=2)

        # --- 3. Bounding box ---
        x1, y1, x2, y2 = [int(v) for v in animal.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        # --- 4. Category label ---
        label = f"#{animal.animal_id} {animal.category} {animal.confidence:.0%}"
        draw_text_with_bg(image, label, (x1, y1 - 6), color=color)

    return image


def draw_eyes(
    image: np.ndarray,
    eye_data: list[dict],
    show_white_ring: bool = True,
    show_inner_dot: bool = True,
) -> np.ndarray:
    """在圖片上繪製眼睛 keypoints。

    === 繪製邏輯 ===
    對每隻動物的每隻眼睛，畫「圓圈框 + 內點」的雙層標記：
    1. 外圈：radius=10 的空心圓（框出眼睛區域，always on）
    2. 白色外環：radius=3 增加可見度（可關閉）
    3. 內點：radius=1 實心圓（可關閉）
    4. "L" / "R" 標籤

    Args:
        eye_data: list of dicts, each with:
            - animal_id: int
            - left_eye: (x, y) or None
            - right_eye: (x, y) or None
        show_white_ring: whether to draw the white ring layer
        show_inner_dot: whether to draw the inner filled dot

    Returns:
        標註後的圖片
    """
    for entry in eye_data:
        color = get_color(entry["animal_id"])

        for eye_key, label in [("left_eye", "L"), ("right_eye", "R")]:
            point = entry.get(eye_key)
            if point is None:
                continue

            img_h, img_w = image.shape[:2]
            x = max(10, min(int(point[0]), img_w - 11))
            y = max(10, min(int(point[1]), img_h - 11))

            # Outer circle frame (marks the eye region)
            cv2.circle(image, (x, y), 10, color, thickness=2)
            # White ring for visibility
            if show_white_ring:
                cv2.circle(image, (x, y), 3, (255, 255, 255), thickness=2)
            # Inner filled dot
            if show_inner_dot:
                cv2.circle(image, (x, y), 1, color, thickness=cv2.FILLED)
            # Label "L" or "R"
            draw_text_with_bg(
                image, label, (x + 10, y + 4),
                font_scale=FONT_SCALE_EYE, color=color,
            )

    return image


def draw_distances(
    image: np.ndarray,
    intra_distances: list[dict] | None = None,
    inter_distances: list[dict] | None = None,
) -> np.ndarray:
    """在圖片上繪製距離測量線和數值。

    === 繪製邏輯 ===
    Intra-animal (雙眼距離):
      - 同一隻動物的左眼到右眼畫一條實線
      - 線的中點標註距離（pixels）

    Inter-animal (右眼距離):
      - 兩隻動物的右眼之間畫一條虛線
      - 線的中點標註距離（pixels）
      - 用白色線條，跟 intra 的彩色線區分

    Args:
        intra_distances: list of dicts with left_eye, right_eye, distance_px, animal_id
        inter_distances: list of dicts with eye_a, eye_b, distance_px, animal_a_id, animal_b_id
    """
    # Sanity check icon colors (BGR)
    SANITY_COLORS = {
        "PASS": (0, 200, 0),      # green
        "WARNING": (0, 200, 255),  # yellow
        "FAIL": (0, 0, 255),       # red
    }
    SANITY_ICONS = {"PASS": "v", "WARNING": "!", "FAIL": "x"}

    # --- Intra-animal: solid colored lines ---
    if intra_distances:
        for d in intra_distances:
            color = get_color(d["animal_id"])
            pt1 = (int(d["left_eye"][0]), int(d["left_eye"][1]))
            pt2 = (int(d["right_eye"][0]), int(d["right_eye"][1]))

            cv2.line(image, pt1, pt2, color, thickness=2)

            # Build cumulative label with all available layers:
            #   none:   "64.8 px"
            #   fast:   "64.8 px | corr: 72.3 px"
            #   metric: "64.8 px | corr: 72.3 px | 0.19m | v cat IOD"
            label = f"{d['distance_px']:.1f} px"
            label_color = color

            corrected_px = d.get("depth_corrected_px")
            metric_m = d.get("metric_distance_m")

            if corrected_px is not None:
                label += f" | corr: {corrected_px:.1f} px"

            if metric_m is not None:
                label += f" | {metric_m:.2f}m"

                check = d.get("sanity_check_result")
                category = d.get("category", "")
                if check and check in SANITY_ICONS:
                    icon = SANITY_ICONS[check]
                    label += f" | {icon} {category} IOD"
                    label_color = SANITY_COLORS.get(check, color)

            # Place label below the eye line, offset perpendicular to avoid
            # covering the bounding box / eye markers
            label_x, label_y = _offset_label(pt1, pt2, offset_px=30)
            draw_text_with_bg(
                image, label, (label_x, label_y),
                font_scale=FONT_SCALE_DISTANCE, color=label_color,
            )

    # --- Inter-animal: dashed white lines ---
    if inter_distances:
        for d in inter_distances:
            pt1 = (int(d["eye_a"][0]), int(d["eye_a"][1]))
            pt2 = (int(d["eye_b"][0]), int(d["eye_b"][1]))

            _draw_dashed_line(image, pt1, pt2, color=(255, 255, 255), thickness=2)

            # Build label: "#0-#1 R-eye: 295.6 px | 2.41m"
            label_text = f"{d['distance_px']:.1f} px"
            corrected_px = d.get("depth_corrected_px")
            metric_m = d.get("metric_distance_m")
            if corrected_px is not None:
                label_text += f" | corr: {corrected_px:.1f} px"
            if metric_m is not None:
                label_text += f" | {metric_m:.2f}m"

            pair_label = f"#{d['animal_a_id']}-#{d['animal_b_id']} R-eye: {label_text}"
            label_x, label_y = _offset_label(pt1, pt2, offset_px=25)
            draw_text_with_bg(
                image, pair_label, (label_x, label_y),
                font_scale=FONT_SCALE_DISTANCE, color=(255, 255, 255),
            )

    return image


def _offset_label(
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    offset_px: int = 30,
) -> tuple[int, int]:
    """Compute a label position offset perpendicular to the line pt1→pt2.

    Places the label below the midpoint of the line (perpendicular direction),
    so it doesn't cover the eyes or bounding boxes.
    If the line is nearly horizontal, offsets downward.
    If the line is nearly vertical, offsets to the right.
    """
    mid_x = (pt1[0] + pt2[0]) // 2
    mid_y = (pt1[1] + pt2[1]) // 2

    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = np.sqrt(dx * dx + dy * dy)

    if length < 1:
        return mid_x, mid_y + offset_px

    # Perpendicular direction (rotated 90 degrees)
    perp_x = -dy / length
    perp_y = dx / length

    # Always push label downward (positive y direction)
    if perp_y < 0:
        perp_x, perp_y = -perp_x, -perp_y

    label_x = int(mid_x + perp_x * offset_px)
    label_y = int(mid_y + perp_y * offset_px)

    return label_x, label_y


def _draw_dashed_line(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    dash_length: int = 10,
    gap_length: int = 8,
) -> None:
    """Draw a dashed line between two points.

    OpenCV 沒有內建虛線功能，所以我們手動計算等距的線段。
    做法是把 pt1→pt2 的向量等分成多段，交替畫線和留空。
    """
    x1, y1 = pt1
    x2, y2 = pt2
    total_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if total_length == 0:
        return

    # Unit direction vector
    dx = (x2 - x1) / total_length
    dy = (y2 - y1) / total_length

    segment_length = dash_length + gap_length
    num_segments = int(total_length / segment_length)

    for i in range(num_segments + 1):
        start_dist = i * segment_length
        end_dist = min(start_dist + dash_length, total_length)

        sx = int(x1 + dx * start_dist)
        sy = int(y1 + dy * start_dist)
        ex = int(x1 + dx * end_dist)
        ey = int(y1 + dy * end_dist)

        cv2.line(image, (sx, sy), (ex, ey), color, thickness)


def visualize_results(
    seg_result: SegmentationResult,
    eye_data: list[dict] | None = None,
    intra_distances: list[dict] | None = None,
    inter_distances: list[dict] | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Complete visualization pipeline: draw all layers on the image.

    === 呼叫順序 ===
    1. draw_segmentation — mask + bbox + label (always)
    2. draw_eyes — eye keypoints (if eye_data provided)
    3. draw_distances — measurement lines (if distance data provided)
    4. Save to file (if output_path provided)

    這個函式設計成「給什麼畫什麼」— 目前只有 segmentation 結果，
    就只畫 mask 和 bbox。等 Step 4 做完 eye detection，
    傳入 eye_data 就會多畫眼睛。Step 5 做完就會畫距離線。

    Args:
        seg_result: Step 3 的 segmentation 結果（必須）
        eye_data: Step 4 的眼睛座標（可選）
        intra_distances: Step 5 的雙眼距離（可選）
        inter_distances: Step 5 的右眼間距離（可選）
        output_path: 輸出圖片路徑（可選，None 則不存檔）

    Returns:
        annotated image (BGR numpy array)
    """
    if seg_result.raw_image is None:
        raise ValueError("SegmentationResult has no raw_image — cannot visualize")

    # Work on a copy to keep the original intact
    canvas = seg_result.raw_image.copy()

    # Layer 1: Segmentation (always drawn)
    canvas = draw_segmentation(canvas, seg_result)

    # Layer 2: Eye keypoints (optional)
    if eye_data:
        canvas = draw_eyes(canvas, eye_data)

    # Layer 3: Distance measurements (optional)
    if intra_distances or inter_distances:
        canvas = draw_distances(canvas, intra_distances, inter_distances)

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)
        logger.info("Saved annotated image to %s", output_path)

    return canvas
