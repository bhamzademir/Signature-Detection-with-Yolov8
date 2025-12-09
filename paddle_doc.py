#!/usr/bin/env python3
# A4 -> YOLO ID model -> ID crop -> akıllı büyütme -> PaddleOCR

import os
import json
import time
import logging
from typing import List, Dict, Any

import numpy as np
import cv2

from paddleocr import PaddleOCR
from ultralytics import YOLO

print(">>> paddle_doc.py import edildi")  # DEBUG


####################################
# LOGGING
####################################
# force=True: başka paketlerin ayarlarını ez
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    force=True
)

logger = logging.getLogger(__name__)


####################################
# YOLO MODEL (ID CARD DETECTOR)
####################################
# Kendi model yolunu yaz:
ID_MODEL_PATH = "runs/detect/id_card_train7/weights/best.pt"
yolo_model = YOLO(ID_MODEL_PATH)


####################################
# ID DETECT + CROP
####################################
def detect_id_and_crop(img_bgr: np.ndarray, pad_ratio: float = 0.05) -> np.ndarray:
    """
    A4 üzerinde ID kartı bulan YOLO modelini kullanıp,
    en büyük kutuyu seçer ve biraz pad ekleyerek crop döner.
    """
    h, w = img_bgr.shape[:2]
    logger.info(f"[ID] Page size: {w}x{h}")

    logger.info("[ID] Running YOLO detection...")
    results = yolo_model(img_bgr, verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        logger.warning("[ID] No ID card detected on page, using full image.")
        return img_bgr

    boxes = results[0].boxes.xyxy.cpu().numpy()  # (N,4)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[idx]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    logger.info(f"[ID] bbox with pad: ({x1}, {y1}) - ({x2}, {y2})")

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        logger.warning("[ID] Empty crop, fallback to full image.")
        return img_bgr

    return crop


def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=(255,255,255))

####################################
# AKILLI BÜYÜTME
####################################
def smart_enlarge(crop_bgr: np.ndarray,
                  target_min_side: int = 900,
                  max_scale: float = 4.0) -> np.ndarray:
    """
    Eğer ID crop'u küçükse büyüt, büyükse dokunma.
    Örnek: max(w,h) < 900 ise, en uzun kenarı 900 olacak şekilde ölçekle.
    """
    h, w = crop_bgr.shape[:2]
    longest = max(w, h)
    logger.info(f"[SCALE] Original crop size: {w}x{h}")

    if longest >= target_min_side:
        logger.info("[SCALE] Crop big enough, no resize.")
        return crop_bgr

    scale = min(target_min_side / longest, max_scale)
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.info(f"[SCALE] Resizing crop {w}x{h} -> {new_w}x{new_h}")

    resized = cv2.resize(
        crop_bgr, (new_w, new_h),
        interpolation=cv2.INTER_LANCZOS4
    )
    return resized


####################################
# OCR INITIALIZATION
####################################
def init_ocr() -> PaddleOCR:
    det_dir = "models/PP-OCRv5_server_det_infer"
    rec_dir = "models/PP-OCRv5_server_rec_infer"

    logger.info("Initializing PaddleOCR v2.7...")

    t0 = time.time()
    ocr = PaddleOCR(
        det_model_dir=det_dir,
        rec_model_dir=rec_dir,
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        det_limit_side_len=2048,
        det_limit_type='max',
        det_db_thresh=0.25,
        det_db_box_thresh=0.7,
    )
    logger.info(f"Models loaded in {time.time() - t0:.2f}s")
    return ocr


####################################
# OCR RUN
####################################
def run_ocr(ocr: PaddleOCR, image_path: str) -> List[Any]:
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return []

    t0 = time.time()
    result = ocr.ocr(image_path, cls=True)
    logger.info(
        f"OCR finished in {time.time() - t0:.2f}s, "
        f"boxes={len(result[0]) if result else 0}"
    )
    return result[0] if result else []


####################################
# VISUALIZATION
####################################
def draw_ocr_results(image_path: str, ocr_lines: List[Any], out_path: str) -> None:
    image = cv2.imread(image_path)

    for line in ocr_lines:
        box = np.array(line[0]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)

        text = line[1][0]
        score = line[1][1]
        cv2.putText(
            image,
            f"{text} ({score:.2f})",
            tuple(box[0][0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    cv2.imwrite(out_path, image)
    logger.info(f"Saved OCR visualization -> {out_path}")


####################################
# JSON BUILD
####################################
def build_json(image_path: str, ocr_lines: List[Any]) -> Dict[str, Any]:
    dt_polys, texts, rec_scores = [], [], []

    for line in ocr_lines:
        dt_polys.append(np.array(line[0]).tolist())
        texts.append(line[1][0])
        rec_scores.append(float(line[1][1]))

    return {
        "input": image_path,
        "dt_polys": dt_polys,
        "texts": texts,
        "rec_scores": rec_scores
    }


####################################
# MAIN
####################################
def main():
    print(">>> main() started")  # DEBUG

    input_path = "dataset/images/train_deskewed/file_280289_page2.jpg"
    out_dir = "ocr_outputs_old"
    os.makedirs(out_dir, exist_ok=True)

    logger.info("STARTING PIPELINE...")

    # 0) A4'i oku
    page = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if page is None:
        logger.error(f"Cannot read image: {input_path}")
        return

    # 1) YOLO ile ID crop'u bul
    id_crop = detect_id_and_crop(page, pad_ratio=0.05)

    """deskewed = deskew(id_crop)
    id_crop = deskewed"""

    # 2) Küçükse büyüt
    id_crop_big = smart_enlarge(id_crop, target_min_side=900)

    # 3) OCR için kaydet
    crop_path = os.path.join(out_dir, "id_processed_" + os.path.basename(input_path))
    cv2.imwrite(crop_path, id_crop_big)
    logger.info(f"[ID] Saved processed ID crop -> {crop_path}")

    # 4) OCR
    ocr = init_ocr()
    ocr_lines = run_ocr(ocr, crop_path)

    

    # 5) JSON SAVE
    out_json = build_json(crop_path, ocr_lines)
    json_path = os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(crop_path))[0] + ".json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved JSON -> {json_path}")

    # 6) DRAW OCR RESULT
    vis_path = os.path.join(out_dir, f"ocr_{os.path.basename(crop_path)}")
    draw_ocr_results(crop_path, ocr_lines, vis_path)

    logger.info("DONE ✔")


if __name__ == "__main__":
    print(">>> __main__ guard triggered")  # DEBUG
    main()
