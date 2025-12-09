#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Belgeyi YOLO ile bul, büyüt ve PaddleOCR (v2.7.x) ile oku.
Hamza için hazırlandı.
"""

import os, cv2, time, json, logging
from typing import List, Dict, Any, Tuple
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

####################################
# AYARLAR
####################################
INPUT_PATH          = "dataset/images/train_deskewed2/file_276747_page1.jpg"
OUT_DIR             = "doc_ocr_outputs"
YOLO_WEIGHTS        = "best.pt"           # eğittiğin model
TARGET_CLASS        = None                # örn: "id_card" / "document" / "signature"; None -> en büyük kutu
CONF_THRESH         = 0.10
IOU_THRESH          = 0.45
PAD_RATIO           = 0.06                # crop etrafına % padding
MIN_WIDTH_TARGET    = 1400                # crop genişliği bundan küçükse büyüt (dokümanlar için iyi başlangıç)
APPLY_ENHANCE       = True                # kontrast, keskinlik, adaptive threshold

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

####################################
# MODELLER
####################################
yolo_model = YOLO(YOLO_WEIGHTS)

def init_ocr() -> PaddleOCR:
    det_dir = "models/PP-OCRv5_server_det_infer"
    rec_dir = "models/PP-OCRv5_server_rec_infer"
    logging.info("Initializing PaddleOCR v2.x ...")
    t0 = time.time()
    ocr = PaddleOCR(
        det_model_dir=det_dir,
        rec_model_dir=rec_dir,
        use_angle_cls=True,
        lang="en",
        use_gpu=False
        # rec_char_dict_path="models/tr_char_dict.txt",
    )
    logging.info(f"OCR loaded in {time.time()-t0:.2f}s")
    return ocr

####################################
# YARARLI ARAÇLAR
####################################
def clamp(v, lo, hi): return max(lo, min(hi, v))

def expand_box(xyxy: np.ndarray, img_w: int, img_h: int, pad_ratio: float) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = map(float, xyxy)
    w = x2-x1; h = y2-y1
    px = w*pad_ratio; py = h*pad_ratio
    x1 = int(clamp(x1-px, 0, img_w-1))
    y1 = int(clamp(y1-py, 0, img_h-1))
    x2 = int(clamp(x2+px, 0, img_w-1))
    y2 = int(clamp(y2+py, 0, img_h-1))
    return x1,y1,x2,y2

def upscale_if_needed(img: np.ndarray, min_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= min_width:
        return img
    scale = min_width / float(w)
    new_size = (int(w*scale), int(h*scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

def enhance_doc(img: np.ndarray) -> np.ndarray:
    # YUV ile hafif kontrast, ardından unsharp mask ve adaptive threshold'u harmanla
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    img_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # unsharp
    blur = cv2.GaussianBlur(img_eq, (0,0), sigmaX=1.2)
    sharp = cv2.addWeighted(img_eq, 1.4, blur, -0.4, 0)

    # gri + adaptive thresh ile hafif binarizasyon (OCR deteksiyonu kolaylaşır)
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    thr = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

    # ikisini harmanla (metin köşelerinde threshold, diğer yerlerde keskin görsel)
    mix = cv2.addWeighted(sharp, 0.6, thr, 0.4, 0)
    return mix

####################################
# YOLO İNFERENCE → EN İYİ BELGE KUTUSU
####################################
def detect_document_box(image_bgr: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """
    Dönüş: (xyxy), class_id, score
    """
    h, w = image_bgr.shape[:2]
    t0 = time.time()
    res = yolo_model.predict(image_bgr, conf=CONF_THRESH, iou=IOU_THRESH, imgsz=1280, verbose=False)
    logging.info(f"YOLO inference: {time.time()-t0:.2f}s")

    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        raise RuntimeError("Belge kutusu bulunamadı.")

    boxes = res[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls   = boxes.cls.cpu().numpy().astype(int)
    conf  = boxes.conf.cpu().numpy()

    names = yolo_model.names
    # sınıf filtresi: önce isme göre, yoksa en büyük alan
    idx_candidates = list(range(len(xyxy)))
    if TARGET_CLASS is not None:
        idx_candidates = [i for i,c in enumerate(cls) if names.get(int(c)) == TARGET_CLASS]
        if not idx_candidates:
            logging.warning(f"TARGET_CLASS='{TARGET_CLASS}' bulunamadı, en büyük kutu seçilecek.")
            idx_candidates = list(range(len(xyxy)))

    # adaylar içinde en büyük alanlı kutuyu seç
    areas = [(i, (xyxy[i][2]-xyxy[i][0])*(xyxy[i][3]-xyxy[i][1])) for i in idx_candidates]
    best_i = max(areas, key=lambda t: t[1])[0]

    return xyxy[best_i], int(cls[best_i]), float(conf[best_i])

####################################
# OCR ÇALIŞTIR
####################################
def run_ocr(ocr: PaddleOCR, image_bgr: np.ndarray) -> List[Any]:
    # paddleocr numpy BGR/RGB fark etmiyor; güvenlisi dosya yolu ama bellekten de işleyebilir:
    # Bu nedenle geçici PNG yazıp okuyacağız (stabilite için).
    tmp_path = os.path.join(OUT_DIR, "_tmp_ocr.png")
    cv2.imwrite(tmp_path, image_bgr)
    result = ocr.ocr(tmp_path, cls=True)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return result[0] if result else []

####################################
# ANA AKIŞ
####################################
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Görseli yükle
    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {INPUT_PATH}")
    H, W = img.shape[:2]

    # 2) YOLO ile belge kutusunu bul
    xyxy, cls_id, score = detect_document_box(img)
    x1,y1,x2,y2 = expand_box(xyxy, W, H, PAD_RATIO)

    # YOLO overlay kaydet
    overlay = img.copy()
    cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 3)
    name = yolo_model.names.get(cls_id, str(cls_id))
    cv2.putText(overlay, f"{name}:{score:.2f}", (int(x1), max(25,int(y1)-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "overlay.jpg"), overlay)

    # 3) Crop + büyütme + iyileştirme
    crop = img[y1:y2, x1:x2].copy()
    crop_up = upscale_if_needed(crop, MIN_WIDTH_TARGET)
    crop_final = enhance_doc(crop_up) if APPLY_ENHANCE else crop_up
    cv2.imwrite(os.path.join(OUT_DIR, "crop.jpg"), crop_final)

    # 4) OCR
    ocr = init_ocr()
    t0 = time.time()
    lines = run_ocr(ocr, crop_final)
    logging.info(f"OCR lines: {len(lines)}; time: {time.time()-t0:.2f}s")

    # 5) OCR görselleştirme (crop üzerinde)
    vis = crop_final.copy()
    for line in lines:
        poly = np.array(line[0]).astype(np.int32).reshape((-1,1,2))
        txt  = line[1][0]; sc = float(line[1][1])
        cv2.polylines(vis, [poly], True, (0,255,0), 2)
        cv2.putText(vis, f"{txt} ({sc:.2f})", tuple(poly[0][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "ocr_vis.jpg"), vis)

    # 6) JSON çıktısı: crop ve tam-resim koordinatlarıyla
    dt_polys_crop, dt_polys_full, texts, rec_scores = [], [], [], []
    for line in lines:
        poly_crop = np.array(line[0], dtype=np.float32)   # crop koordinatı
        poly_full = poly_crop.copy()
        poly_full[:,:,0] += x1
        poly_full[:,:,1] += y1

        dt_polys_crop.append(poly_crop.tolist())
        dt_polys_full.append(poly_full.tolist())
        texts.append(line[1][0])
        rec_scores.append(float(line[1][1]))

    out_json = {
        "input": INPUT_PATH,
        "yolo": {
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "class_id": int(cls_id),
            "class_name": yolo_model.names.get(int(cls_id), str(cls_id)),
            "score": float(score),
            "pad_ratio": PAD_RATIO
        },
        "preprocess": {
            "min_width_target": MIN_WIDTH_TARGET,
            "applied_enhance": APPLY_ENHANCE
        },
        "crop_path": os.path.join(OUT_DIR, "crop.jpg"),
        "overlay_path": os.path.join(OUT_DIR, "overlay.jpg"),
        "ocr_vis_path": os.path.join(OUT_DIR, "ocr_vis.jpg"),
        "ocr": {
            "dt_polys_crop": dt_polys_crop,
            "dt_polys_full": dt_polys_full,
            "texts": texts,
            "rec_scores": rec_scores
        }
    }
    json_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(INPUT_PATH))[0] + "_doc_ocr.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved JSON -> {json_path}")
    logging.info("Done.")

if __name__ == "__main__":
    main()
