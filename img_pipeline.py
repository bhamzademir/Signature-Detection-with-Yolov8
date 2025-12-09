#!/usr/bin/env python3

# ocr_kv_processor_fixed_clean.py

import os
import re
import json
import time
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from typing import Dict, List, Any, Tuple
from functools import lru_cache
from pdf2image import convert_from_path
import math

# ======================
# Config
# ======================
REC_DICT_PATH = None
SCORE_THRESH = 0.50 # Yüksek güvenilirlikteki sonuçları koru

# ======================
# OCR init (predict)
# ======================
# KRİTİK: GPU'nun kullanılmasını sağlıyoruz
ocr = PaddleOCR(
    ocr_version='PP-OCRv5',
    use_textline_orientation=True,
    text_det_box_thresh=0.25,
    text_det_unclip_ratio=1.5,
    text_det_thresh=0.20,
    text_det_limit_side_len=960,
    text_det_limit_type='min',

    cpu_threads=8,
    **({"rec_char_dict_path": REC_DICT_PATH} if REC_DICT_PATH else {})
)

# -------- Label ve normalizasyon kuralları --------
LABELS = {
    "foreigner_id_number": ["yabancikimlikno", "foreigneridentitynumber", "kimlikno"],
    "given_name": ["adi", "adiname", "name", "adi/name"],
    "surname": ["soyadi", "soyadisurname", "surname", "soyadi/surname"],
    "nationality": ["uyrugu", "uyrugunationality", "nationality"],
    "date_of_birth": ["dogumtarihi", "dateofbirth", "dob"],
    "place_of_birth": ["dogumyeri", "placeofbirth"],
    "province_of_residence": ["ikametili", "provinceofresidence"],
    "document_number": ["serino", "documentno", "serialno", "seri"],
    "mother_name": ["anneadi", "mothersname", "anaadi"],
    "father_name": ["babaadi", "fathersname"],
    "expiry_date": ["duedate", "bitistarihi", "expirydate"]
}
DATE_RX = re.compile(r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b")
ID11_RX = re.compile(r"\b\d{11}\b")
DOC_RX  = re.compile(r"\b([A-Z]{3}\s?\d{6,})\b")

# -------- Türkçe karakter dönüştürme + normalizasyon --------
TR_MAP = str.maketrans({
    "ı":"i","İ":"i","ş":"s","Ş":"s","ğ":"g","Ğ":"g",
    "ü":"u","Ü":"u","ö":"o","Ö":"o","ç":"c","Ç":"c"
})

def _norm(s: str) -> str:
    s = (s or "").translate(TR_MAP).lower()
    return re.sub(r"[^a-z0-9]", "", s)

# ====== FUZZY LABEL MATCH AYARLARI ======
FUZZY_LABEL_SIM = 0.78
FUZZY_PREFIX_BOOST = True

# ——— fuzzy oranı ———
try:
    from rapidfuzz.distance import Levenshtein as _lev
    def levenshtein_ratio(a: str, b: str) -> float:
        return 1.0 - _lev.normalized_distance(a, b)
except Exception:
    from difflib import SequenceMatcher
    def levenshtein_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

# LABELS -> normalize edilmiş varyant havuzu
_LABEL_VARIANTS_NORM: Dict[str, List[str]] = {
    field: [_norm(v) for v in variants]
    for field, variants in LABELS.items()
}
_LABEL_ALL: List[Tuple[str, str]] = [(f, nv) for f, arr in _LABEL_VARIANTS_NORM.items() for nv in arr]

@lru_cache(maxsize=4096)
def _best_label_match_norm(nt: str) -> Tuple[str, float]:
    """
    nt: önceden _norm uygulanmış metin.
    Döndürür: (field, score). Score 0–1 arası benzerlik.
    """
    if not nt:
        return "", 0.0
    for field, arr in _LABEL_VARIANTS_NORM.items():
        for nv in arr:
            if nt.startswith(nv) or nv in nt:
                return field, 1.0

    best_field, best_score = "", 0.0
    for field, nv in _LABEL_ALL:
        s = levenshtein_ratio(nt, nv)
        if FUZZY_PREFIX_BOOST and len(nt) >= 3:
            s = max(s, levenshtein_ratio(nt[:len(nv)], nv))
        if s > best_score:
            best_field, best_score = field, s
    return best_field, best_score

def _is_label(text: str) -> Tuple[str, bool]:
    nt = _norm(text)
    field, score = _best_label_match_norm(nt)
    return (field, True) if score >= FUZZY_LABEL_SIM else ("", False)

def _as_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        return arr[:, :, :3]
    return arr

def _tolist_box(box):
    try:
        return np.asarray(box).astype(float).tolist()
    except Exception:
        return box

def _center(box: List[List[float]]) -> Tuple[float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (sum(xs)/4.0, sum(ys)/4.0)

# YENİ: İki kutu arasındaki mesafeyi hesapla
def _distance(box1, box2):
    c1 = _center(box1)
    c2 = _center(box2)
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# -------- PDF to Image conversion --------
def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    try:
        images = convert_from_path(pdf_path, dpi=300)
        print(f"Converted PDF to {len(images)} image(s)")
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return []

def get_image_for_processing(file_path: str) -> List[Tuple[Image.Image, str]]:
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.pdf':
        images = convert_pdf_to_images(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        return [(img, f"{base_name}_page{i}") for i, img in enumerate(images)]
    else:
        # İYİLEŞTİRME 2: PIL Image doğrudan 'ocr.predict'e veriliyor, gereksiz numpy dönüşümü kaldırıldı
        img = Image.open(file_path).convert("RGB") # RGB'ye zorla
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        return [(img, base_name)]

# -------- OCR predict parse --------
def _parse_predict_dict(d: dict, score_thresh: float = 0.0):
    texts  = d.get("rec_texts")  or []
    scores = d.get("rec_scores") or []
    boxes  = (d.get("rec_polys") or d.get("rec_boxes") or d.get("dt_polys") or [])
    n = min(len(texts), len(scores), len(boxes))
    out = []
    for i in range(n):
        sc = float(scores[i])
        if sc < score_thresh:
            continue
        # Benzersiz ID ekle (etiket/değer ayrımı için kritik)
        out.append({
            "id": i, # Benzersiz ID
            "text": texts[i],
            "confidence": sc,
            "box": _tolist_box(boxes[i])
        })
    return out

# -------- OCR predict running --------
def process_image_predict(image: Image.Image, score_thresh: float = 0.0):
    start = time.time()
    # PaddleOCR, numpy array bekliyor
    result = ocr.predict(np.array(image))

    if isinstance(result, list) and result and isinstance(result[0], dict):
        result = result[0]

    if not isinstance(result, dict):
        return {
            "image_size": list(image.size),
            "results": [],
            "processing_time": time.time() - start,
            "num_detected": 0,
            "note": "predict beklenen dict yerine farklı tip döndürdü"
        }

    processed = _parse_predict_dict(result, score_thresh=score_thresh)
    return {
        "image_size": list(image.size),
        "processing_time": time.time() - start,
        "num_detected": len(processed),
        "results": processed
    }

# -------- Key–Value çıkarımı --------
def extract_key_values(ocr_json: Dict[str, Any]) -> Dict[str, str]:
    items = ocr_json.get("results", [])
    if not items:
        return {}

    img_w, img_h = ocr_json.get("image_size", [1920, 1080])

    # Tüm öğelere merkez koordinat ve normalize metin ekle
    enriched = []
    for it in items:
        box = it.get("box")
        if not box:
            continue
        cx, cy = _center(box)
        enriched.append({**it, "cx": cx, "cy": cy, "nt": _norm(it["text"])})

    # Etiketleri bul. Sadece etiketlere 'field' ve 'is_label' bayrağı ekle
    labels = []
    label_ids = set() # Etiket olarak sınıflandırılanların ID'leri
    for it in enriched:
        field, ok = _is_label(it["text"])
        if ok:
            labels.append({**it, "field": field})
            label_ids.add(it["id"])

    # Etiketler için aday değeri seç:
    # İYİLEŞTİRME 3: Konumsal toleransları görsel boyutuna göre ayarla
    x_tol = img_w * 0.20           # X hizası toleransı (kolon hizası)
    y_tol_same_line = img_h * 0.02 # Aynı satır toleransı

    result: Dict[str, str] = {}

    for lab in labels:
        lx, ly, lbox = lab["cx"], lab["cy"], lab["box"]

        # 1. Sağdaki adayları bul (aynı satırda en yakın)
        right_candidates = []
        for it in enriched:
            # KRİTİK DÜZELTME: Adayın bir etiket OLMADIĞINDAN emin ol
            if it["id"] in label_ids:
                continue

            # Konumsal kontrol: Sağda ve aynı satırda (Y toleransı dahilinde)
            if it["cx"] > lx and (abs(it["cy"] - ly) <= y_tol_same_line):
                right_candidates.append(it)

        # En yakın sağdaki adayı seç
        right_candidates.sort(key=lambda it: it["cx"] - lx) # X koordinatına göre sırala
        text_val = right_candidates[0]["text"] if right_candidates else ""

        # 2. Sağda bulunamazsa: Etiketin ALTINDA aynı kolon hizasında en yakın adayı bul
        if not text_val:
            down_candidates = []
            for it in enriched:
                # KRİTİK DÜZELTME: Adayın bir etiket OLMADIĞINDAN emin ol
                if it["id"] in label_ids:
                    continue

                # Konumsal kontrol: Aşağıda ve aynı kolon hizasında (X toleransı dahilinde)
                if it["cy"] > ly and (abs(it["cx"] - lx) <= x_tol):
                    down_candidates.append(it)

            # En yakın aşağıdaki adayı seç
            # Mesafe ve Y farkına göre sırala
            down_candidates.sort(key=lambda it: (it["cy"] - ly, abs(it["cx"] - lx)))
            text_val = down_candidates[0]["text"] if down_candidates else ""

        # 3. Alan bazlı normalizasyon ve veri doğrulama
        f = lab["field"]
        raw = (text_val or "").strip()

        if raw:
            # İYİLEŞTİRME 4: Alan bazlı veri doğrulama
            if f in ["given_name", "surname", "mother_name", "father_name", "nationality", "place_of_birth"]:
                # Alfabetik veya isme benzeyen değerleri koru
                clean_raw = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]', '', raw).strip()
                if len(clean_raw) / len(raw.replace(' ', '')) < 0.5: # Çoğu karakter rakam/sembol ise atla
                    continue
                raw = clean_raw

            # Regex tabanlı normalizasyon
            if f == "foreigner_id_number":
                m = ID11_RX.search(raw.replace(" ", ""))
                if m: raw = m.group(0)
            elif f == "date_of_birth" or f == "expiry_date":
                m = DATE_RX.search(raw)
                if m: raw = m.group(1)
            elif f == "document_number":
                m = DOC_RX.search(raw.replace(" ", ""))
                if m: raw = m.group(1).replace(" ", "")

            if raw:
                result[f] = raw

    # Post-processing for high-confidence ID pattern
    id_num_val = None
    for it in enriched:
        raw_text = (it["text"] or "").strip().replace(" ", "")
        m = ID11_RX.search(raw_text)
        if m:
            id_num_val = m.group(0)
            break

    if id_num_val:
        # ID numarasını atama (yüksek öncelik)
        result["foreigner_id_number"] = id_num_val
        # Diğer alanlara yanlışlıkla atanmış ID'yi temizle
        for field_name in ["surname", "date_of_birth", "document_number"]:
             if result.get(field_name) == id_num_val:
                result[field_name] = ""

    # (Opsiyonel) Belge tipi
    header_join = " ".join([it["text"] for it in enriched[:30]]).lower()
    if "residence permit" in header_join or "ikamet izni" in header_join:
        result.setdefault("document_type", "İkamet İzni Belgesi")
    if "mavi kart" in header_join or "mawikart" in header_join:
        result.setdefault("document_type", "Mavi Kart")

    return result

# -------- I/O --------
def save_json(payload: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Örnek dosya adını Colab ortamınıza yüklediğiniz dosya adı ile değiştirin
    input_file = "dataset/images/train_deskewed/file_263350_page2.jpg"
    out_dir = "outputs"

    print(f"Processing: {input_file}")
    print("=" * 50)

    start_total = time.time()
    images = get_image_for_processing(input_file)

    if not images:
        print("Error: Could not load images from file")
        exit(1)

    all_results = []
    for img, page_id in images:
        print(f"\nProcessing {page_id}...")
        ocr_json_path = os.path.join(out_dir, f"{page_id}.json")
        kv_json_path  = os.path.join(out_dir, f"{page_id}_kv.json")

        # OCR
        ocr_out = process_image_predict(img, score_thresh=SCORE_THRESH)
        save_json(ocr_out, ocr_json_path)

        # Key-Value extraction
        kv = extract_key_values(ocr_out)
        save_json(kv, kv_json_path)

        print(f"Processing completed in {ocr_out['processing_time']:.2f} s")
        print(f"Detected {ocr_out['num_detected']} text regions")
        print(f"Key-Value JSON: {kv_json_path}")
        print(json.dumps(kv, ensure_ascii=False, indent=2))

        all_results.append({"page": page_id, "data": kv})

    end_total = time.time()
    print("\n" + "=" * 50)
    print(f"All pages processed successfully! Total time: {end_total - start_total:.2f} s")