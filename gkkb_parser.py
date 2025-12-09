#!/usr/bin/env python3
# gkkb_parser.py — TR GKKB / İkamet İzni / Mavi Kart gibi belgeler için
# OCR -> satır tabanlı + alan (proximity) eşleştirmeli parser.
# Hem PaddleOCR liste formatını hem dict-of-arrays JSON'u destekler.

import json, argparse, sys, re
from typing import List, Dict, Any, Tuple, Optional
from difflib import get_close_matches

# =========================
# Parametreler
# =========================
Y_TOL = 16               # satır gruplama dikey toleransı (px)
CENTER_Y_TOL = 20        # label-değer merkez Y hizası toleransı
RIGHT_MIN_DX = 6         # aynı satırda "sağında" kabul min delta-x
FUZZY_CUTOFF = 0.70      # label fuzzy eşiği
MERGE_GAP = 8            # aynı satırda bitişik kutuları birleştirme boşluğu

MIN_VALUE_SCORE = 0.7  # minimum değer OCR skoru (şimdilik kullanılmıyor)

# =========================
# Label sözlüğü
# =========================
LABELS_CANON = {
    # TR (GKKB / İkamet)
    "yabanci kimlik no": "foreigner_id",
    "yabanci kimlik": "foreigner_id",
    "yabanci kimlik no.": "foreigner_id",
    "yabancikimlikno": "foreigner_id",

    "blue card ": "blue card", 
    "mavi kart": "blue card",

    "çalışma izni": "work permit",
    "calisma izni": "work permit",

    "gecici koruma kimlik": "temporary protection id",
    "geçici koruma kimlik": "temporary protection id",
    "gecici koruma" : "temporary protection id",
    "geçici koruma" : "temporary protection id",

    "ikamet izni": "residence",
    "ikametgah": "residence",
    "ikamet": "residence",

    "adi": "given_name",
    "ad": "given_name",
    "ADI": "given_name",
    "given names": "given_name",
    "soyadi": "surname",
    "soyad": "surname",

    "baba adi": "father_name",
    "babaadi": "father_name",
    "baba ad": "father_name",

    "ana adi": "mother_name",
    "anaadi": "mother_name",

    "dogum tarihi": "birth_date",
    "dogum tarifi": "birth_date",
    "dogum tarh": "birth_date",
    "dogum tarih": "birth_date",
    "dogum yeri": "birth_place",

    "medenidurum": "marital_status",
    "medeni durumu": "marital_status",
    "medeni durumu:": "marital_status",

    "aile no": "family_no",
    "aile no.": "family_no",

    "uyrugu": "nationality",
    "uyruk": "nationality",

    "kayit tarihi": "reg_date",
    "kayit tarih": "reg_date",
    "kayit tarhi": "reg_date",
    "kayıt tarihi": "reg_date",

    "seri": "document_no",
    "seri no": "document_no",
    "seri no.": "document_no",
    "belge no": "document_no",
    "ikamet ili": "province",
    "ikamet il": "province",

    # EN (iki dilli başlıklar için)
    "foreigner identity number": "foreigner_id",
    "name": "given_name",
    "surname": "surname",
    "father name": "father_name",
    "mother name": "mother_name",
    "date of birth": "birth_date",
    "place of birth": "birth_place",
    "nationality": "nationality",
    "document no": "document_no",
    "document number": "document_no",
    "province of residence": "province",

    # Mavi Kart varyantları
    "kimlik no": "identity_no",
    "kimlikno": "identity_no",
}

DATE_RX = re.compile(r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})')
NUM_RX  = re.compile(r'\d+')
ID11_RX = re.compile(r'\b\d{11}\b')

# =========================
# Geometri yardımcıları
# =========================
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

# =========================
# Metin normalize & label match
# =========================
def _norm(s: str) -> str:
    s = s.strip().lower()
    s = (s.replace("ı","i").replace("ö","o").replace("ü","u")
           .replace("ş","s").replace("ç","c").replace("ğ","g"))
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def match_label(raw_text: str) -> Optional[str]:
    t = _norm(raw_text)
    if t in LABELS_CANON:
        return LABELS_CANON[t]
    # iki dilli başlıkları böl (Adi/Name, Nationality | Date of Birth vb.)
    parts = re.split(r'[\/\|\:\-]+', raw_text)
    for p in parts:
        pn = _norm(p)
        if pn in LABELS_CANON:
            return LABELS_CANON[pn]
    # fuzzy
    cand = get_close_matches(t, LABELS_CANON.keys(), n=1, cutoff=FUZZY_CUTOFF)
    if cand:
        return LABELS_CANON[cand[0]]
    for p in parts:
        pn = _norm(p)
        cand = get_close_matches(pn, LABELS_CANON.keys(), n=1, cutoff=FUZZY_CUTOFF)
        if cand:
            return LABELS_CANON[cand[0]]
    return None

INLINE_SEP_RX = re.compile(r'[|:/\-]+')

def extract_inline_value(raw_text: str, field: str) -> Optional[str]:
    """
    Tek kutuda 'Label | Değer' (veya 'Label : Değer') formatını yakalar.
    Örn: 'Yabanci Kimlik No | 99464414416' -> '99464414416'
         'Anne Adi | NAJAH | LU'          -> 'NAJAH LU'
    """
    parts = [p.strip() for p in INLINE_SEP_RX.split(raw_text) if p.strip()]
    if not parts:
        return None

    # parçalar içinde label konumunu bul
    label_idx = None
    for i, p in enumerate(parts):
        if match_label(p) == field:
            label_idx = i
            break
    if label_idx is None:
        return None

    # label'dan sonraki parçaları birleştir (bir sonraki label'a kadar)
    right_parts = []
    for p in parts[label_idx+1:]:
        if match_label(p):  # tekrar bir label geldi -> dur
            break
        right_parts.append(p)

    if not right_parts:
        return None

    candidate = " ".join(right_parts).strip()

    # alan tipine göre ufak temizleme
    if field in ("birth_date", "reg_date"):
        m = DATE_RX.search(candidate)
        candidate = m.group(1) if m else candidate
        candidate = candidate.replace('-', '.').replace('/', '.')
    elif field in ("foreigner_id", "identity_no", "family_no"):
        # 11 haneli (YKN) ya da genel sayısal temizleme
        m = ID11_RX.search(candidate)
        if field in ("foreigner_id", "identity_no"):
            candidate = m.group(0) if m else candidate
        elif field == "family_no":
            m2 = NUM_RX.search(candidate)
            candidate = m2.group(0) if m2 else candidate
    elif field == "document_no":
        candidate = re.sub(r'\s+', ' ', candidate)
        candidate = re.sub(r'^(seri|series?)\s+', '', candidate, flags=re.I).strip()

    return candidate if candidate else None

# =========================
# Satır içi kutu birleştirme
# =========================
def _try_merge_same_line(chunks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not chunks: return []
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

# =========================
# OCR normalize (çoklu format)
# =========================
def normalize_ocr(obj: Any) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []

    # dict-of-arrays formatı
    if isinstance(obj, dict) and "dt_polys" in obj and "texts" in obj:
        polys = obj["dt_polys"]
        texts = obj["texts"]
        scores = obj.get("rec_scores", [1.0]*len(texts))
        for box, text, sc in zip(polys, texts, scores):
            cx, cy = _bbox_center(box)
            out.append({"text": str(text), "box": box, "score": float(sc), "cx": cx, "cy": cy})
        out = _try_merge_same_line(out)
        out = sorted(out, key=lambda x: (x['cy'], _bbox_left_right(x['box'])[0]))
        return out

    # PaddleOCR liste / dict formatları
    if isinstance(obj, list):
        for item in obj:
            if (isinstance(item, (list, tuple)) and len(item) == 2 and
                isinstance(item[0], (list, tuple)) and len(item[0]) == 4):
                box = item[0]
                text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                score = item[1][1] if (isinstance(item[1], (list, tuple)) and len(item[1]) > 1) else 1.0
                cx, cy = _bbox_center(box)
                out.append({"text": str(text), "box": box, "score": float(score), "cx": cx, "cy": cy})
            elif isinstance(item, dict) and "text" in item and "box" in item:
                box = item["box"]
                cx, cy = _bbox_center(box)
                sc = float(item.get("score", 1.0))
                out.append({"text": str(item["text"]), "box": box, "score": sc, "cx": cx, "cy": cy})

    out = _try_merge_same_line(out)
    out = sorted(out, key=lambda x: (x['cy'], _bbox_left_right(x['box'])[0]))
    return out

# =========================
# Satır gruplama
# =========================
def group_lines(items: List[Dict[str,Any]], y_tol:int=Y_TOL) -> List[List[Dict[str,Any]]]:
    lines: List[List[Dict[str,Any]]] = []
    cur: List[Dict[str,Any]] = []
    last_y: Optional[float] = None
    for it in items:
        if last_y is None or abs(it['cy'] - last_y) <= y_tol:
            cur.append(it)
        else:
            lines.append(sorted(cur, key=lambda x: _bbox_left_right(x['box'])[0]))
            cur = [it]
        last_y = it['cy']
    if cur:
        lines.append(sorted(cur, key=lambda x: _bbox_left_right(x['box'])[0]))
    return lines

# =========================
# Label -> Değer eşleştirme
# =========================
def find_value_for_label(label_item, line_items, all_items, lines, line_idx, field: Optional[str]=None):
    # 0) Inline case
    inline = extract_inline_value(label_item['text'], field) if field else None
    if inline:
        return inline

    lx1, lx2 = _bbox_left_right(label_item['box'])
    lcy = label_item['cy']

    # 1) Aynı satırda SAĞINDA — birden çok yakını birleştir (|'ları atla)
    cands = []
    for it in line_items:
        if it is label_item:
            continue
        # düşük skorlu (genelde Arapça / gürültü) kutuları değer adayı olarak alma
        if it.get("score", 1.0) < MIN_VALUE_SCORE:
            continue
        x1, _ = _bbox_left_right(it['box'])
        if x1 - lx2 >= RIGHT_MIN_DX and abs(it['cy'] - lcy) <= CENTER_Y_TOL:
            cands.append(it)


    if cands:
        cands.sort(key=lambda z: _bbox_left_right(z['box'])[0])
        texts = []
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
        # 🔽 EKLEYECEĞİN SATIR BU 🔽
        return cands[0]["text"].strip()

    # 2) ALT SATIR: 1–3 satır aşağı, yatay örtüşme **veya** merkez-x yakınlığı
    lcx = (lx1 + lx2) / 2.0
    for look_ahead in (1, 2, 3):
        j = line_idx + look_ahead
        if j >= len(lines): break
        below = lines[j]
        # Eğer satırda tek non-label öğe varsa direkt onu al
        non_labels = [
            it for it in below
            if not match_label(it['text'])
            and it.get("score", 1.0) >= MIN_VALUE_SCORE
        ]
        if len(non_labels) == 1:
            return non_labels[0]['text'].strip()

        below_cands = []
        for it in non_labels:
            cx = sum(_bbox_left_right(it['box'])) / 2.0
            overlap_ok = _horiz_overlap_ratio(label_item['box'], it['box']) >= 0.15  # 0.25 -> 0.15
            center_ok  = abs(cx - lcx) <= 180                                     # 120 -> 180
            if overlap_ok or center_ok:
                dy = abs(it['cy'] - lcy)
                below_cands.append((dy, abs(cx-lcx), it))
        if below_cands:
            below_cands.sort(key=lambda t: (t[0], t[1]))
            return below_cands[0][2]['text'].strip()

    # 3) Fallback: sayfa genelinde sağ-hizada en yakın
    near = []
    for it in all_items:
        if it is label_item:
            continue
        if it.get("score", 1.0) < MIN_VALUE_SCORE:
            continue
        x1, _ = _bbox_left_right(it['box'])
        dx = x1 - lx2
        if dx >= RIGHT_MIN_DX and abs(it['cy'] - lcy) <= CENTER_Y_TOL:
            near.append((dx, abs(it['cy'] - lcy), it))
    if near:
        near.sort(key=lambda t: (t[0], t[1]))
        return near[0][2]['text'].strip()
    return None


# =========================
# Postprocess
# =========================
def postprocess_value(field: str, value: str) -> str:
    v = value.strip()
    if field in ("birth_date", "reg_date"):
        m = DATE_RX.search(v)
        if m: v = m.group(1)
        v = v.replace('-', '.').replace('/', '.')
    elif field in ("foreigner_id", "identity_no"):
        m = ID11_RX.search(v)
        print("matching 11 id", m)
        if m: v = m.group(0)
    elif field in ("family_no",):
        m = NUM_RX.search(v)
        if m: v = m.group(0)
    elif field in ("given_name", "surname", "father_name", "mother_name"):
        # Örn: "الاسم MOHAMAD" / "الكنية AKOURA" gibi satırlarda
        # baştaki Arapça kısımı at, ilk Latin harften sonrası kalsın.
        m = re.search(r'[A-Za-zÇĞİÖŞÜçğıöşü]', v)
        if m:
            v = v[m.start():]
        v = re.sub(r'\s+', ' ', v).strip()
    elif field == "document_no":
        v = re.sub(r'\s*\|\s*', ' ', v)   # "SERI | A01 | N717030" -> "SERI A01 N717030"
        v = re.sub(r'\s+', ' ', v).strip()
        # "SERI " öne geldiyse temizle
        v = re.sub(r'^(seri|series?)\s+', '', v, flags=re.I)
    return v

def fix_swapped_fields(result: Dict[str,str]) -> None:
    """
    Bazı GKKB kartlarında OCR / layout kayması nedeniyle:
      - nationality alanına tarih ya da damga parçası,
      - family_no alanına ise "SURIYE" gibi uyruk değeri yazılabiliyor.

    Eğer:
      * nationality içinde rakam varsa (örn. "35:12200" gibi çöp),
      * family_no tamamen büyük harflerden oluşan bir kelimeyse ("SURIYE", "IRAK" vb.)
    o zaman bunların yer değiştirdiğini varsayıp düzelt.
    Ayrıca reg_date alanı tarih içermiyorsa ve eski nationality içinde
    tarih pattern'i varsa, onu reg_date olarak kullan.
    """
    nat = result.get("nationality")
    fam = result.get("family_no")
    reg = result.get("reg_date")

    def is_all_caps_word(s: str) -> bool:
        return bool(re.fullmatch(r"[A-ZÇĞİÖŞÜ ]+", s))

    if nat and any(ch.isdigit() for ch in nat) and fam and is_all_caps_word(fam):
        # nationality bozuk, family_no ise muhtemelen uyruk
        result["nationality"] = fam
        if (not reg or not DATE_RX.search(reg)) and DATE_RX.search(nat or ""):
            # damga satırının içinde tarih varsa onu kayıt tarihi olarak çek
            result["reg_date"] = DATE_RX.search(nat).group(1).replace('-', '.').replace('/', '.')


# =========================
# Heuristic: Aile No'dan sonra Nationality (GKKB)
# =========================
def infer_nationality_after_family_no(lines: List[List[Dict[str,Any]]]) -> Optional[str]:
    fam_idx = None
    for i, line in enumerate(lines):
        for it in line:
            if match_label(it['text']) == "family_no":
                fam_idx = i
                break
        if fam_idx is not None:
            break
    if fam_idx is not None and fam_idx + 1 < len(lines):
        cand = " ".join(it["text"].strip() for it in lines[fam_idx+1])
        cand_clean = re.sub(r'\s+', ' ', cand).strip()
        if 2 <= len(cand_clean) <= 22 and cand_clean.upper() == cand_clean and re.search(r'[A-Z]', cand_clean):
            return cand_clean
    return None

# =========================
# Parse pipeline
# =========================
def parse_items(items: List[Dict[str,Any]], debug: bool=False) -> Dict[str,Any]:
    lines = group_lines(items, Y_TOL)
    if debug:
        for li, line in enumerate(lines):
            print(f"[line {li:02d}] " + " | ".join([it["text"] for it in line]), file=sys.stderr)

    result: Dict[str,str] = {}

    for li, line in enumerate(lines):
        for it in line:
            field = match_label(it['text'])
            if not field:
                continue
            if field in result and result[field]:
                continue
            val = find_value_for_label(it, line, items, lines, li, field)
            if val:
                result[field] = postprocess_value(field, val)

    # 11 haneli fallback (kimlik/yk no)
    # 11 haneli kimlik override: foreigner_id varsa ama 11 haneli değilse düzelt
    cur_ykn = result.get("foreigner_id") or result.get("identity_no")
    if not cur_ykn or not ID11_RX.fullmatch(str(cur_ykn)):
        for it in items:
            m = ID11_RX.search(it['text'])
            if m:
                # GKKB'de öncelik foreigner_id
                result["foreigner_id"] = m.group(0)
                break

    # Bazı alanların OCR / satır kaymasından dolayı birbirine karışma ihtimaline
    # karşı ufak bir düzeltme (özellikle nationality / family_no / reg_date).
        fix_swapped_fields(result)

        if "nationality" not in result:
            nat = infer_nationality_after_family_no(lines)
            if nat:
                result["nationality"] = nat

        return {"document_type": "TR_GKKB", "fields": result}

    if "nationality" not in result:
        nat = infer_nationality_after_family_no(lines)
        if nat:
            result["nationality"] = nat

    return {"document_type": "TR_GKKB", "fields": result}

# =========================
# (Opsiyonel) Auto-orient — sadece --image modunda
# =========================
def auto_orient_image(path: str):
    try:
        import cv2, numpy as np
        from paddleocr import PaddleOCR
    except Exception:
        raise

    def _rotate(img, k):
        if k == 0: return img
        if k == 1: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if k == 2: return cv2.rotate(img, cv2.ROTATE_180)
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _largest_rect_warp(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return img
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.15*img.shape[0]*img.shape[1]:
            pts = approx.reshape(4,2).astype("float32")
            s = pts.sum(axis=1); diff = (pts[:,0]-pts[:,1])
            rect = [pts[s.argmin()], pts[diff.argmin()], pts[s.argmax()], pts[diff.argmax()]]
            rect = np.array(rect, dtype="float32")
            (tl,tr,br,bl) = rect
            wA = np.linalg.norm(br-bl); wB = np.linalg.norm(tr-tl)
            hA = np.linalg.norm(tr-br); hB = np.linalg.norm(tl-bl)
            maxW = int(max(wA,wB)); maxH = int(max(hA,hB))
            dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(img, M, (maxW, maxH))
        return img

    import cv2
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Görüntü okunamadı: {path}")
    img = _largest_rect_warp(img)

    ocr = PaddleOCR(lang='tr', ocr_version='PP-OCRv5', use_angle_cls=True, det=True, rec=True)

    def score(img_np):
        res = ocr.ocr(img_np, cls=True)
        if not res or not isinstance(res[0], list): return 0.0
        sc = 0.0
        for it in res[0]:
            try: sc += float(it[1][1])
            except Exception: pass
        return sc

    best, best_sc = img, -1
    for k in (0,1,2,3):
        cand = _rotate(img, k)
        sc = score(cand)
        if sc > best_sc:
            best, best_sc = cand, sc
    return best

# =========================
# I/O & main
# =========================
def run_from_json(path: str, debug: bool=False) -> Dict[str,Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = normalize_ocr(raw)
    if debug:
        print(f"Loaded items: {len(items)}", file=sys.stderr)
    return parse_items(items, debug=debug)

def run_from_image(path: str, debug: bool=False) -> Dict[str,Any]:
    from paddleocr import PaddleOCR
    import cv2
    best_img = auto_orient_image(path)

    ocr = PaddleOCR(lang='tr', ocr_version='PP-OCRv5', use_angle_cls=True, det=True, rec=True)
    res = ocr.ocr(best_img, cls=True)
    if isinstance(res, list) and len(res)>0 and isinstance(res[0], list):
        res = res[0]
    items = normalize_ocr(res)

    # Arapça ikinci geçiş (varsa)
    try:
        ocr_ar = PaddleOCR(lang='ar', ocr_version='PP-OCRv5', use_angle_cls=True, det=True, rec=True)
        res_ar = ocr_ar.ocr(best_img, cls=True)
        if isinstance(res_ar, list) and len(res_ar)>0 and isinstance(res_ar[0], list):
            res_ar = res_ar[0]
        items += normalize_ocr(res_ar)
        items = normalize_ocr(items)
    except Exception:
        pass

    if debug:
        print(f"Loaded items (image): {len(items)}", file=sys.stderr)
    return parse_items(items, debug=debug)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="PaddleOCR JSON yolu (dict-of-arrays da desteklenir)")
    ap.add_argument("--image", help="Görüntü yolu (auto-orient aktif)")
    ap.add_argument("--debug", action="store_true", help="Satır dökümü stderr'e yazılır")
    args = ap.parse_args()

    if not args.json and not args.image:
        print("Kullanım: --json ocr.json  veya  --image belge.jpg  [--debug]", file=sys.stderr)
        sys.exit(2)

    if args.json:
        out = run_from_json(args.json, debug=args.debug)
    else:
        out = run_from_image(args.image, debug=args.debug)

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
