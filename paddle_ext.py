#!/usr/bin/env python3
# A4 -> YOLO ID model -> ID crop -> akıllı büyütme -> PaddleOCR -> GKKB Parser + Flask API

import os
import json
import time
import logging
import sys
import re
import argparse
import uuid
from typing import List, Dict, Any, Tuple, Optional
from difflib import get_close_matches

import numpy as np
import cv2

from paddleocr import PaddleOCR
from ultralytics import YOLO

from flask import Flask, request, jsonify

print(">>> ocr_app.py başlatıldı")

####################################
# LOGGING
####################################
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    force=True
)

logger = logging.getLogger(__name__)

####################################
# YOLO MODEL (ID CARD DETECTOR)
####################################
#ID_MODEL_PATH = "runs/detect/id_card_train7/weights/best.pt"
ID_MODEL_PATH = "models/best.pt"
try:
    yolo_model = YOLO(ID_MODEL_PATH)
except Exception as e:
    logger.warning(f"YOLO modeli yüklenemedi: {e}")
    yolo_model = None

MIN_OCR_SCORE = 0.80

def detect_id_and_crop(img_bgr: np.ndarray, pad_ratio: float = 0.05) -> np.ndarray:
    """
    A4 içinden ID card bounding box'ını bulup kırp.
    pad_ratio: kutunun etrafına ekstra boşluk koyma oranı (%)
    """
    if yolo_model is None:
        return img_bgr

    H, W = img_bgr.shape[:2]
    logger.info(f"[ID] Input page size: {W}x{H}")

    results = yolo_model.predict(
        source=img_bgr,
        verbose=False,
        conf=0.25
    )

    if not results or len(results[0].boxes) == 0:
        logger.warning("[ID] YOLO hiçbir bbox bulamadı, tüm sayfayı döndürüyorum.")
        return img_bgr

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(scores))
    x1, y1, x2, y2 = boxes[best_idx]
    conf = scores[best_idx]
    logger.info(f"[ID] Best bbox: {(x1, y1, x2, y2)} conf={conf:.3f}")

    w = x2 - x1
    h = y2 - y1
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio

    x1p = max(int(x1 - pad_x), 0)
    y1p = max(int(y1 - pad_y), 0)
    x2p = min(int(x2 + pad_x), W)
    y2p = min(int(y2 + pad_y), H)

    crop = img_bgr[y1p:y2p, x1p:x2p].copy()
    logger.info(f"[ID] Cropped ID size: {crop.shape[1]}x{crop.shape[0]}")
    return crop


def smart_enlarge(crop_bgr: np.ndarray,
                  target_min_side: int = 900) -> np.ndarray:
    """
    ID crop küçükse, OCR için daha okunabilir olsun diye ölçeklendir.
    """
    h, w = crop_bgr.shape[:2]
    min_side = min(w, h)
    if min_side >= target_min_side:
        logger.info(f"[ENLARGE] No need to enlarge, min_side={min_side}")
        return crop_bgr

    scale = target_min_side / float(min_side)
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.info(f"[ENLARGE] scale={scale:.2f}, new size={new_w}x{new_h}")
    enlarged = cv2.resize(crop_bgr, (new_w, new_h),
                          interpolation=cv2.INTER_CUBIC)
    return enlarged


####################################
# PADDLE OCR INIT & RUN
####################################
_OCR_INSTANCE: Optional[PaddleOCR] = None

def init_ocr() -> PaddleOCR:
    """
    PaddleOCR instance. Dil ve parametreler ihtiyaca göre değiştirilebilir.
    """
    det_dir = "models/PP-OCRv5_server_det_infer"
    rec_dir = "models/PP-OCRv5_server_rec_infer"
    logger.info("Initializing PaddleOCR v2.7...")
    ocr = PaddleOCR(
        det_model_dir=det_dir,
        rec_model_dir=rec_dir,
        use_angle_cls=True,
        lang="en",
        #use_gpu=False,
        det_limit_side_len=2048,
        det_limit_type='max',
        det_db_thresh=0.25,
        det_db_box_thresh=0.7,
    )
    return ocr

def get_ocr() -> PaddleOCR:
    """
    OCR motorunu lazy-load şekilde tek sefer initialize eder.
    Flask request'lerinde tekrar tekrar load etmez.
    """
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        _OCR_INSTANCE = init_ocr()
    return _OCR_INSTANCE

def run_ocr(ocr: PaddleOCR, image_path: str) -> List[Any]:
    """
    PaddleOCR çıktısını ham liste formatında döndürür.
    """
    logger.info(f"[OCR] Running on {image_path} ...")
    start = time.time()
    result = ocr.ocr(image_path, cls=True)
    logger.info(f"[OCR] Done in {time.time() - start:.2f} sec")
    
    if not result:
        return []
    return result[0]


def build_json(image_path: str, ocr_lines: List[Any]) -> Dict[str, Any]:
    """
    PaddleOCR çıktısını dict-of-arrays formuna çevirir.
    """
    dt_polys, texts, rec_scores = [], [], []

    if ocr_lines:
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


# ============================================================================
# BÖLÜM 2: GELİŞMİŞ PARSER (GKKB / İKAMET)
# ============================================================================

Y_TOL = 16
CENTER_Y_TOL = 20
RIGHT_MIN_DX = 6
FUZZY_CUTOFF = 0.70
MERGE_GAP = 30
MIN_VALUE_SCORE = 0.70

INLINE_SEP_RX = re.compile(r'[|:/\-]+')
ID11_RX       = re.compile(r'\b\d{11}\b')
NUM_RX        = re.compile(r'\d+')
DATE_RX       = re.compile(r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})')
DIGIT_RX      = re.compile(r'\d+')

LABEL_MAP = {
    "adi": "given_name",
    "ad": "given_name",
    "ADI": "given_name",
    "name": "given_name",
    "given names": "given_name",
    "soyadi": "surname",
    "soyad": "surname",
    "surname": "surname",

    "baba adi": "father_name",
    "babaadi": "father_name",
    "baba ad": "father_name",

    "ana adi": "mother_name",
    "anaadi": "mother_name",
    "ana ad": "mother_name",
    "anne adi": "mother_name",
    "anneadi": "mother_name",
    "anne ad": "mother_name",

    "dogum tarihi": "birth_date",
    "doğum tarihi": "birth_date",
    "birth date": "birth_date",
    "dogum tari": "birth_date",
    "date of birth": "birth_date",

    "dogum yeri": "birth_place",
    "doğum yeri": "birth_place",
    "birth place": "birth_place",

    "uyruk": "nationality",
    "nationality": "nationality",
    "nationalıt": "nationality",
    "natonal": "nationality",

    "serino": "document_no",
    "seri no": "document_no",
    "seri no.": "document_no",
    "document no": "document_no",
    "doc no": "document_no",

    "kimlik no": "foreigner_id",
    "yabancı kimlik no": "foreigner_id",
    "yabanci kimlik no": "foreigner_id",
    "foreigner identity number": "foreigner_id",
    "tc kimlik no": "foreigner_id",

    "kayıt tarihi": "reg_date",
    "kayit tarihi": "reg_date",
    "reg. date": "reg_date",

    "aile sıra no": "family_no",
    "aile sira no": "family_no",
    "aile sirano": "family_no",
    "family no": "family_no",

    "medeni hal": "marital_status",
    "medeni durumu": "marital_status",
    "marital status": "marital_status",

    "gecici koruma kimlik belgesi": "temporary protection id",
    "geçici koruma kimlik belgesi": "temporary protection id",
}

def _bbox_center(b: List[List[float]]) -> Tuple[float,float]:
    xs = [p[0] for p in b]; ys = [p[1] for p in b]
    return (sum(xs)/4.0, sum(ys)/4.0)

def _bbox_left_right(b: List[List[float]]) -> Tuple[float,float]:
    xs = [p[0] for p in b]
    return (min(xs), max(xs))

def _bbox_top_bottom(b: List[List[float]]) -> Tuple[float,float]:
    ys = [p[1] for p in b]
    return (min(ys), max(ys))

def _horiz_overlap_ratio(a_box, b_box) -> float:
    a_l, a_r = _bbox_left_right(a_box)
    b_l, b_r = _bbox_left_right(b_box)
    inter = max(0.0, min(a_r, b_r) - max(a_l, b_l))
    denom = max((a_r - a_l), (b_r - b_l), 1.0)
    return inter / denom

def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("İ", "i").replace("I", "i")
    s = s.lower()
    tr_map = str.maketrans("çğıöşü", "cgiosu")
    s = s.translate(tr_map)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def match_label(raw: str) -> Optional[str]:
    norm = normalize_text(raw)
    if norm in LABEL_MAP:
        return LABEL_MAP[norm]

    parts = re.split(r'[:|/\-]+', raw)
    for part in parts:
        p_norm = normalize_text(part)
        if p_norm in LABEL_MAP:
            return LABEL_MAP[p_norm]

    keys = list(LABEL_MAP.keys())
    candidates = get_close_matches(norm, keys, n=1, cutoff=FUZZY_CUTOFF)
    if candidates:
        return LABEL_MAP[candidates[0]]
    return None

def extract_inline_value(raw_text: str, field: str) -> Optional[str]:
    text = raw_text.strip()
    tokens = text.split()
    if len(tokens) < 2:
        return None

    # Label başta, değer sonda
    for i in range(1, len(tokens)):
        cand_label = " ".join(tokens[:i])
        if match_label(cand_label) == field:
            val_part = " ".join(tokens[i:]).strip()

            clean_val = re.sub(r'^[:|/\-]+\s*', '', val_part)
            if match_label(clean_val):
                return None

            if re.search(r'(Name|Surname|Date|Birth|Identity|Number|No)\b', clean_val, re.IGNORECASE):
                if not any(char.isdigit() for char in clean_val):
                    return None

            if not clean_val:
                return None

            return clean_val

    # Değer başta, label sonda (nadir)
    for i in range(1, 4):
        if len(tokens) <= i:
            break
        cand_label = " ".join(tokens[-i:])
        if match_label(cand_label) == field:
            val = " ".join(tokens[:-i]).strip()
            if match_label(val):
                return None
            return val

    return None

def extract_id_11(text: str) -> Optional[str]:
    m = ID11_RX.search(text)
    if m:
        return m.group(0)
    return None

def extract_digits(text: str, length: Optional[int]=None) -> Optional[str]:
    digits = "".join(DIGIT_RX.findall(text))
    if not digits:
        return None
    if length and len(digits) != length:
        return None
    return digits

def is_line_label(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    if re.search(r'[/|:-]', text) and len(text.split()) <= 6:
        parts = re.split(r'[:|/\-]+', text)
        for p in parts:
            if match_label(p):
                return True
    if match_label(text):
        return True
    low = text.lower()
    short_label_words = ['adi', 'soyad', 'dogum', 'tarihi', 'belge', 'no', 'uyruk', 'calisma', 'izni', 'anne', 'baba', 'name', 'surname', 'date']
    for w in short_label_words:
        if re.search(r'\b' + re.escape(w) + r'\b', low):
            if any(ch.isdigit() for ch in text):
                continue
            return True
    return False

def _try_merge_same_line(chunks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not chunks:
        return []
    chunks = sorted(chunks, key=lambda x: _bbox_left_right(x['box'])[0])
    merged = [chunks[0]]
    for cur in chunks[1:]:
        prev = merged[-1]
        if abs(prev['cy'] - cur['cy']) <= CENTER_Y_TOL:
            if _bbox_left_right(cur['box'])[0] - _bbox_left_right(prev['box'])[1] <= MERGE_GAP:
                prev['text'] = (prev['text'] + ' ' + cur['text']).strip()
                l1, r1 = _bbox_left_right(prev['box'])
                l2, r2 = _bbox_left_right(cur['box'])
                t1, b1 = _bbox_top_bottom(prev['box'])
                t2, b2 = _bbox_top_bottom(cur['box'])
                prev['box'] = [[min(l1,l2), min(t1,t2)],
                               [max(r1,r2), min(t1,t2)],
                               [max(r1,r2), max(b1,b2)],
                               [min(l1,l2), max(b1,b2)]]
                prev['cx'], prev['cy'] = _bbox_center(prev['box'])
                continue
        merged.append(cur)
    return merged

def normalize_ocr(obj: Any) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []

    def add_if_valid(text, box, score):
        if score < MIN_OCR_SCORE:
            return
        if len(text) < 2 and not text.isdigit():
            return
        cx, cy = _bbox_center(box)
        out.append({"text": str(text), "box": box, "score": float(score), "cx": cx, "cy": cy})

    if isinstance(obj, dict) and "dt_polys" in obj and "texts" in obj:
        polys = obj["dt_polys"]
        texts = obj["texts"]
        scores = obj.get("rec_scores", [1.0]*len(texts))
        for box, text, sc in zip(polys, texts, scores):
            add_if_valid(text, box, sc)
    elif isinstance(obj, list):
        for line in obj:
            if isinstance(line, list):
                for item in line:
                    if len(item) >= 2:
                        box = item[0]
                        text = item[1][0]
                        sc = float(item[1][1])
                        add_if_valid(text, box, sc)

    out = _try_merge_same_line(out)
    out = sorted(out, key=lambda x: (x['cy'], _bbox_left_right(x['box'])[0]))
    return out

def group_lines(items: List[Dict[str,Any]], y_tol:int=Y_TOL) -> List[List[Dict[str,Any]]]:
    lines: List[List[Dict[str,Any]]] = []
    if not items:
        return lines

    current_line: List[Dict[str,Any]] = [items[0]]
    last_y = items[0]["cy"]

    for item in items[1:]:
        if abs(item["cy"] - last_y) <= y_tol:
            current_line.append(item)
        else:
            current_line = sorted(current_line, key=lambda x: _bbox_left_right(x["box"])[0])
            lines.append(current_line)
            current_line = [item]
        last_y = item["cy"]

    current_line = sorted(current_line, key=lambda x: _bbox_left_right(x["box"])[0])
    lines.append(current_line)
    return lines

def find_label_value_pairs(lines: List[List[Dict[str,Any]]]) -> List[Tuple[Dict[str,Any], Dict[str,Any]]]:
    pairs: List[Tuple[Dict[str,Any], Dict[str,Any]]] = []
    for line in lines:
        n = len(line)
        if n < 2:
            continue
        for i in range(n):
            cand_label = line[i]
            field = match_label(cand_label["text"])
            if not field:
                continue

            lx1, lx2 = _bbox_left_right(cand_label["box"])
            lcx, lcy = cand_label["cx"], cand_label["cy"]
            best_val, best_dx = None, None

            for j in range(i+1, n):
                cand_val = line[j]
                if is_line_label(cand_val["text"]):
                    continue

                vx1, vx2 = _bbox_left_right(cand_val["box"])
                vcy = cand_val["cy"]
                dx = vx1 - lx2
                if dx < RIGHT_MIN_DX:
                    continue
                if abs(vcy - lcy) > CENTER_Y_TOL:
                    continue
                if _horiz_overlap_ratio(cand_label["box"], cand_val["box"]) > 0.45:
                    continue

                if best_dx is None or dx < best_dx:
                    best_dx = dx
                    best_val = cand_val
            if best_val:
                pairs.append((cand_label, best_val))
    return pairs

def find_value_for_label(
    label_item: Dict[str, Any],
    line_items: List[Dict[str, Any]],
    all_items: List[Dict[str, Any]],
    lines: List[List[Dict[str, Any]]],
    line_idx: int,
    field: Optional[str] = None
) -> Optional[str]:
    inline = extract_inline_value(label_item['text'], field) if field else None
    if inline:
        return inline

    lx1, lx2 = _bbox_left_right(label_item['box'])
    lcy = label_item['cy']

    # 1) Aynı satır sağ
    cands: List[Dict[str, Any]] = []
    for it in line_items:
        if it is label_item:
            continue
        if it.get("score", 1.0) < MIN_VALUE_SCORE:
            continue
        if is_line_label(it['text']):
            continue
        x1, _ = _bbox_left_right(it['box'])
        if x1 - lx2 >= RIGHT_MIN_DX and abs(it['cy'] - lcy) <= CENTER_Y_TOL:
            if _horiz_overlap_ratio(label_item['box'], it['box']) > 0.7:
                continue
            cands.append(it)

    if cands:
        cands.sort(key=lambda z: _bbox_left_right(z['box'])[0])
        texts: List[str] = []
        last_r = lx2
        join_gap = 120 if field == "document_no" else 80
        for it in cands:
            l, r = _bbox_left_right(it['box'])
            if l - last_r <= join_gap:
                tok = it['text'].strip()
                if tok and tok not in {"|", "¦", "/"}:
                    texts.append(tok)
                last_r = r
            else:
                break
        if texts:
            return " ".join(texts)
        return cands[0]["text"].strip()

    # 2) Alt satır
    lcx = (lx1 + lx2) / 2.0
    for look_ahead in (1, 2, 3):
        j = line_idx + look_ahead
        if j >= len(lines):
            break
        below = lines[j]
        non_labels = [it for it in below if not is_line_label(it['text']) and it.get("score", 1.0) >= MIN_VALUE_SCORE]
        
        if not non_labels:
            continue
        if len(non_labels) == 1:
            return non_labels[0]['text'].strip()

        below_cands = []
        for it in non_labels:
            cx = sum(_bbox_left_right(it['box'])) / 2.0
            overlap_ok = _horiz_overlap_ratio(label_item['box'], it['box']) >= 0.15
            center_ok = abs(cx - lcx) <= 180
            if overlap_ok or center_ok:
                dy = abs(it['cy'] - lcy)
                below_cands.append((dy, abs(cx - lcx), it))
        if below_cands:
            below_cands.sort(key=lambda t: (t[0], t[1]))
            return below_cands[0][2]['text'].strip()

    # 3) Fallback: sayfa genel sağ
    near = []
    for it in all_items:
        if it is label_item:
            continue
        if it.get("score", 1.0) < MIN_VALUE_SCORE:
            continue
        if is_line_label(it['text']):
            continue
        x1, _ = _bbox_left_right(it['box'])
        dx = x1 - lx2
        if dx >= RIGHT_MIN_DX and abs(it['cy'] - lcy) <= CENTER_Y_TOL:
            near.append((dx, abs(it['cy'] - lcy), it))
    if near:
        near.sort(key=lambda t: (t[0], t[1]))
        return near[0][2]['text'].strip()

    return None


def parse_items(items: List[Dict[str,Any]], debug: bool=False) -> Dict[str,Any]:
    lines = group_lines(items, y_tol=Y_TOL)
    if debug:
        print("==== LINES DUMP ====", file=sys.stderr)
        for li, line in enumerate(lines):
            print(f"[LINE {li}] text='{' '.join([x['text'] for x in line])}'", file=sys.stderr)

    out_fields: Dict[str, Any] = {}
    all_flat_items = [item for line in lines for item in line]

    # ADIM 1: Klasik Pair
    pairs = find_label_value_pairs(lines)
    for lb, val in pairs:
        field = match_label(lb["text"])
        if field and field not in out_fields:
            out_fields[field] = val["text"].strip()

    # ADIM 2: Dikey Arama
    for li, line in enumerate(lines):
        for it in line:
            field = match_label(it["text"])
            if not field:
                continue
            if field in out_fields and out_fields[field]:
                continue

            val = find_value_for_label(it, line, all_flat_items, lines, li, field)
            if val:
                out_fields[field] = val

    final_fields: Dict[str, Any] = {}

    for field, raw_val in out_fields.items():
        clean_val = raw_val

        if field == "nationality":
            date_match = DATE_RX.search(clean_val)
            if date_match:
                found_date = date_match.group(1)
                clean_val = clean_val.replace(found_date, "").strip()
                if "birth_date" not in out_fields:
                    final_fields["birth_date"] = found_date.replace("-", ".").replace("/", ".")

        if field == "foreigner_id":
            id11 = extract_id_11(raw_val)
            if id11:
                clean_val = id11
            else:
                id11 = extract_id_11(raw_val.replace(" ", ""))
                if id11:
                    clean_val = id11

        elif field in ("family_no", "document_no"):
            digits = extract_digits(raw_val)
            if digits:
                clean_val = digits

        final_fields[field] = clean_val

    # YKN fallback
    if "foreigner_id" not in final_fields:
        for it in all_flat_items:
            id11 = extract_id_11(it["text"].replace(" ", ""))
            if id11:
                final_fields["foreigner_id"] = id11
                break

    fix_swapped_fields(final_fields)

    if "nationality" not in final_fields:
        nat = infer_nationality_after_family_no(lines)
        if nat:
            final_fields["nationality"] = nat

    if "nationality" not in final_fields:
        for it in all_flat_items:
            norm = normalize_text(it["text"])
            if "suriye" in norm:
                final_fields["nationality"] = "SURIYE"
                break
            if "turkiye" in norm:
                final_fields["nationality"] = "TURKIYE"
                break

    return {
        "document_type": "TR_GKKB",
        "fields": final_fields
    }

def fix_swapped_fields(result: Dict[str, Any]) -> None:
    nat = result.get("nationality")
    fam = result.get("family_no")
    reg = result.get("reg_date")
    
    def is_all_caps_word(s: str) -> bool:
        return bool(re.fullmatch(r"[A-ZÇĞİÖŞÜ ]+", s))

    if nat and any(ch.isdigit() for ch in nat) and fam and is_all_caps_word(fam):
        result["nationality"] = fam
        if (not reg or not DATE_RX.search(reg)) and DATE_RX.search(nat or ""):
            dt = DATE_RX.search(nat).group(1)
            result["reg_date"] = dt.replace("-", ".").replace("/", ".")

def infer_nationality_after_family_no(lines: List[List[Dict[str, Any]]]) -> Optional[str]:
    fam_idx = None
    for i, line in enumerate(lines):
        for it in line:
            if match_label(it['text']) == "family_no":
                fam_idx = i
                break
        if fam_idx is not None:
            break

    if fam_idx is not None and fam_idx + 1 < len(lines):
        cand = " ".join(it["text"].strip() for it in lines[fam_idx + 1])
        cand_clean = re.sub(r'\s+', ' ', cand).strip()
        if (2 <= len(cand_clean) <= 22 and cand_clean.upper() == cand_clean and re.search(r'[A-Z]', cand_clean)):
            return cand_clean
    return None

####################################
# FULL PIPELINE
####################################
def run_full_pipeline(input_path: str, out_dir: str = "ocr_full", debug: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    logger.info("FULL PIPELINE START...")

    page = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if page is None:
        raise RuntimeError(f"Cannot read image: {input_path}")

    id_crop = detect_id_and_crop(page, pad_ratio=0.05)
    id_crop_big = smart_enlarge(id_crop, target_min_side=900)
    crop_path = os.path.join(out_dir, "id_processed_" + os.path.basename(input_path))
    cv2.imwrite(crop_path, id_crop_big)
    logger.info(f"[ID] Saved processed ID crop -> {crop_path}")

    ocr = get_ocr()
    ocr_lines = run_ocr(ocr, crop_path)

    out_json = build_json(crop_path, ocr_lines)
    json_path = os.path.join(out_dir, os.path.splitext(os.path.basename(crop_path))[0] + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved OCR JSON -> {json_path}")

    items = normalize_ocr(out_json)
    parsed = parse_items(items, debug=debug)

    result = {
        "crop_path": crop_path,
        "ocr_json_path": json_path,
        "ocr_raw": out_json,
        "parsed": parsed,
    }

    logger.info("FULL PIPELINE DONE ✔")
    return result

####################################
# CLI MODE (opsiyonel)
####################################
def main_cli():
    ap = argparse.ArgumentParser(description="A4 -> ID OCR -> GKKB parser tam pipeline (CLI)")
    ap.add_argument("--image", required=True, help="A4 görüntü yolu")
    ap.add_argument("--out_dir", default="ocr_outputs_final", help="Çıkış klasörü")
    ap.add_argument("--debug", action="store_true", help="Parser debug çıktısı")
    args = ap.parse_args()

    res = run_full_pipeline(args.image, out_dir=args.out_dir, debug=args.debug)
    print(json.dumps(res["parsed"], ensure_ascii=False, indent=2))

####################################
# FLASK APP
####################################
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ocr/scan", methods=["POST"])
def ocr_scan():
    if 'file' not in request.files:
        logger.error("Request içinde 'file' bulunamadı")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            ext = os.path.splitext(file.filename)[1]
            if not ext:
                ext = ".jpg"
            unique_filename = f"{uuid.uuid4()}{ext}"
            temp_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            file.save(temp_path)
            logger.info(f"Dosya kaydedildi: {temp_path}")

            doc_type = request.form.get('docType', 'UNKNOWN')
            logger.info(f"İşlem başlıyor. DocType: {doc_type}")

            result = run_full_pipeline(
                input_path=temp_path,
                out_dir="ocr_outputs",
                debug=False
            )

            # İstersen geçici dosyayı sil:
            # os.remove(temp_path)

            response = {
                "success": True,
                "result": {
                    "fields": result.get("parsed", {}).get("fields", {}),
                    "raw_text": " ".join(result.get("ocr_raw", {}).get("texts", [])),
                    "docType": doc_type
                }
            }
            return jsonify(response)

        except Exception as e:
            logger.error(f"OCR Hatası: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unknown error"}), 400

####################################
# ENTRYPOINT
####################################
if __name__ == "__main__":
    # Eğer komut satırında --image varsa CLI modunda çalış
    if "--image" in sys.argv:
        main_cli()
    else:
        # API modunda Flask'i çalıştır
        app.run(host="0.0.0.0", port=6000, debug=False)
