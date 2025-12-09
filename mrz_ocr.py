from __future__ import annotations
import os 
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io, json, uuid, argparse, logging, re, base64, datetime, time
from typing import Optional, Tuple, Dict, List, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from paddleocr import PaddleOCR

import numpy as np
import cv2
import pdf2image
from fastmrz import FastMRZ

# ---- API (FastAPI)
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Config
# =========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("MRZ-API")

TESSERACT_PATH = "/usr/bin/tesseract"
MAX_SIDE = 1600
DEFAULT_WORKERS = max(1, os.cpu_count() or 2)

# =========================
# GLOBAL MODEL CACHING (FIX)
# =========================
# Bu değişken EasyOCR modelini hafızada tutacak
GLOBAL_EASYOCR_READER = None
GLOBAL_PADDLE_OCR = None

def get_easyocr_reader():
    """
    Modeli sadece bir kez yükler ve global değişkende saklar.
    Sonraki çağrılarda diskten okuma yapmaz, RAM'den getirir.
    """
    global GLOBAL_EASYOCR_READER
    if GLOBAL_EASYOCR_READER is None:
        import easyocr
        log.info("⚡ EasyOCR modeli RAM'e yükleniyor... (Bu işlem süreç başına 1 kez yapılır)")
        # GPU varsa True yapabilirsiniz, yoksa False kalmalı
        GLOBAL_EASYOCR_READER = easyocr.Reader(['tr','en'], gpu=False, verbose=False)
    return GLOBAL_EASYOCR_READER

# =========================
# Errors & helpers
# =========================
class ProcessingError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ProcessingError("ENCODE_FAIL", "PNG encode failed")
    return buf.tobytes()

def imread_gray_bytes(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise ProcessingError("EMPTY_FILE", "Dosya boş")
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        pages = pdf2image.convert_from_bytes(file_bytes, dpi=200)
        if not pages:
            raise ProcessingError("PDF_EMPTY", "PDF sayfa içermiyor")
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2GRAY)
    except Exception as e:
        raise ProcessingError("DECODE_FAIL", f"Görsel/PDF decode başarısız: {e}")
    
def imread_color_bytes(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise ProcessingError("EMPTY_FILE", "Dosya boş")
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pages = pdf2image.convert_from_bytes(file_bytes, dpi=200)
        if not pages:
            raise ProcessingError("PDF_EMPTY", "PDF sayfa içermiyor")
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ProcessingError("DECODE_FAIL", f"Görsel/PDF decode başarısız: {e}")

def resize_max(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)

# =========================
# Preprocess & pipeline
# =========================
def illumination_correction(img, blur_ks=31):
    bg = cv2.GaussianBlur(img, (blur_ks, blur_ks), 0)
    norm = (img.astype(np.float32) / (bg.astype(np.float32)+1e-3)) * 128.0
    return to_uint8(norm)

def clahe(img, clip=2.0, grid=8):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
    return c.apply(img)

def tophat_blackhat(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    toph = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blkh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    enh = cv2.addWeighted(img, 1.0, toph, 1.0, 0)
    enh = cv2.subtract(enh, blkh)
    return enh

def adaptive_binarize(img):
    b1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,15)
    b2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,10)
    return b1, b2

def gamma_correction(img, gamma=1.2):
    norm = img.astype(np.float32)/255.0
    out = np.power(norm, gamma)
    return to_uint8(out*255.0)

def unsharp_mask(img, ksize=0, sigma=1.0, amount=1.5, thresh=0):
    if ksize == 0:
        ksize = int(6*sigma+1) | 1
    blurred = cv2.GaussianBlur(img, (ksize,ksize), sigma)
    sharp = cv2.addWeighted(img, 1+amount, blurred, -amount, 0)
    if thresh > 0:
        low_contrast_mask = np.absolute(img-blurred) < thresh
        np.copyto(sharp, img, where=low_contrast_mask)
    return sharp

def combined_intelligent_pipeline(img):
    illum = illumination_correction(img)
    cl = clahe(illum, 2.0, 8)
    return unsharp_mask(cl, sigma=1.0, amount=1.5)

def try_perspective(img):
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,15)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, False
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) != 4:
        return img, False
    pts = approx.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    maxW = int(max(wA, wB)); maxH = int(max(hA, hB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl,tr,br,bl],dtype=np.float32), dst)
    warped = cv2.warpPerspective(img, M, (maxW,maxH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped, True

def crop_mrz_band(gray):
    h, w = gray.shape[:2]
    roi = gray[int(h*0.55):, :]
    norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
    bw = cv2.adaptiveThreshold(norm,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,35,15)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(max(w//40,10),3)), iterations=1)
    proj = bw.sum(axis=1)
    nz = np.where(proj > 0)[0]
    if nz.size:
        top = max(nz[0]-15, 0)
        bot = min(nz[-1]+15, bw.shape[0]-1)
        band = roi[top:bot, :]
    else:
        band = roi
    scale = 2.0 if min(band.shape[:2]) < 220 else 1.3
    band = cv2.resize(band, (int(band.shape[1]*scale), int(band.shape[0]*scale)), cv2.INTER_CUBIC)
    band = cv2.bitwise_not(cv2.adaptiveThreshold(band,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,10))
    return band

def pipelines(img):
    out = []
    base = resize_max(img, MAX_SIDE)
    out.append(("base", base))
    try:
        intelligent = combined_intelligent_pipeline(base)
        out.append(("intelligent_ensemble", intelligent))
    except Exception as e:
        log.warning(f"Intelligent pipeline error: {e}")
    try:
        band = crop_mrz_band(base)
        out.append(("mrz_crop", band))
        # out.append(("mrz_crop_gamma_1.2", gamma_correction(band, 1.2))) # Fazlalıkları azalttım
    except Exception as e:
        log.warning(f"mrz_crop hata: {e}")
    
    # Çok fazla varyant süreyi uzatabilir, en etkilileri tutalım
    # out.append(("tophat_blackhat", tophat_blackhat(base)))
    b1, _ = adaptive_binarize(base)
    out.append(("bin_gauss", b1))
    
    try:
        warped, ok = try_perspective(base)
        if ok:
            out.append(("perspective", warped))
    except Exception as e:
        log.warning(f"perspective hata: {e}")
    return out

# =========================
# MRZ validators & scoring
# =========================
try:
    from rapidfuzz.distance import Levenshtein as _lev
    def levenshtein_ratio(a: str, b: str) -> float:
        return 1.0 - _lev.normalized_distance(a, b)
except Exception:
    from difflib import SequenceMatcher
    def levenshtein_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

WEIGHTS = [7,3,1]
TD3_L1_MIN_RE = re.compile(r'^P[A-Z<][A-Z]{3}[A-Z0-9<]*$')
TD1_LINE1_RE = re.compile(r'^[ACI][A-Z0-9<][A-Z]{3}[A-Z0-9<]{9}[0-9][A-Z0-9<]{15}$')
TD1_LINE2_RE = re.compile(r'^[0-9]{6}[0-9][MFX<][0-9]{6}[0-9][A-Z]{3}[A-Z0-9<]{11}[0-9]$')
TD1_LINE3_RE = re.compile(r'^[A-Z0-9<]{30}$')

def _mrz_char_value(c: str) -> int:
    if c.isdigit(): return ord(c) - 48
    if 'A' <= c <= 'Z': return ord(c) - 55
    if c == '<': return 0
    return 0

def _compute_cd(field: str) -> str:
    s = 0
    for i, ch in enumerate(field):
        s += _mrz_char_value(ch) * WEIGHTS[i % 3]
    return str(s % 10)

def _normalize_for_field(s: str, digits_only=False):
    s = s.upper().replace(' ', '<').replace('«', '<').replace('‹', '<')
    if digits_only:
        s = (s.replace('O', '0').replace('D', '0').replace('Q', '0')
               .replace('I', '1').replace('L', '1')
               .replace('S', '5').replace('B', '8').replace('Z', '2'))
    return s

def extract_td1_lines(raw_text: str):
    lines = [ln.strip().upper() for ln in raw_text.splitlines() if ln.strip()]
    lines = [''.join(ch for ch in ln if ch.isalnum() or ch == '<') for ln in lines]
    candidates = [ln for ln in lines if 25 <= len(ln) <= 35]
    if len(candidates) >= 3:
        L1, L2, L3 = candidates[-3], candidates[-2], candidates[-1]
    elif len(lines) >= 3:
        L1, L2, L3 = lines[-3], lines[-2], lines[-1]
    else:
        return None
    L1 = (L1).ljust(30, '<')[:30]
    L2 = (L2).ljust(30, '<')[:30]
    L3 = (L3).ljust(30, '<')[:30]
    return L1, L2, L3

def validate_mrz_td1(raw_text: str):
    trip = extract_td1_lines(raw_text)
    if not trip:
        return {"is_valid": False, "checks_passed": 0, "details": {"reason":"no_lines"}, "lines": ("","","")}
    L1, L2, L3 = trip
    doc_code  = L1[0:2]
    issuer    = _normalize_for_field(L1[2:5])
    doc_no    = _normalize_for_field(L1[5:14])
    doc_cd    = L1[14]
    dob       = _normalize_for_field(L2[0:6],  digits_only=True)
    dob_cd    = L2[6]
    sex       = L2[7]
    exp       = _normalize_for_field(L2[8:14], digits_only=True)
    exp_cd    = L2[14]
    nat       = _normalize_for_field(L2[15:18])
    opt2      = _normalize_for_field(L2[18:29])
    comp_cd   = L2[29]

    chk_doc = (_compute_cd(doc_no) == doc_cd)
    chk_dob = (_compute_cd(dob)    == dob_cd)
    chk_exp = (_compute_cd(exp)    == exp_cd)
    composite_str = L1[5:30] + L2[0:7] + L2[8:15] + L2[18:29]
    chk_comp = (_compute_cd(composite_str) == comp_cd)

    checks_passed = sum([chk_doc, chk_dob, chk_exp, chk_comp])
    head_ok = bool(TD1_LINE1_RE.match(L1)) and bool(TD1_LINE2_RE.match(L2)) and bool(TD1_LINE3_RE.match(L3))
    nat_ok  = len(nat) == 3 and all('A' <= c <= 'Z' for c in nat)
    is_valid = head_ok and nat_ok and (checks_passed >= 1)
    return {
        "doc_type": "TD1",
        "is_valid": is_valid,
        "checks_passed": checks_passed,
        "details": {
            "document_number_cd": chk_doc,
            "date_of_birth_cd":   chk_dob,
            "date_of_expiry_cd":  chk_exp,
            "composite_cd":       chk_comp,
            "nat_ok": nat_ok,
            "head_ok": head_ok,
            "issuer": issuer,
            "sex": sex,
            "opt2": opt2
        },
        "lines": (L1, L2, L3)
    }

def _extract_td3_lines(raw_text: str):
    lines = [ln.strip().upper() for ln in raw_text.splitlines() if ln.strip()]
    lines = [''.join(ch for ch in ln if ch.isalnum() or ch == '<') for ln in lines]
    candidates = [ln for ln in lines if 30 <= len(ln) <= 50]
    if len(candidates) >= 2:
        L1, L2 = candidates[-2], candidates[-1]
    elif len(lines) >= 2:
        L1, L2 = lines[-2], lines[-1]
    else:
        return None
    L1 = (_normalize_for_field(L1)).ljust(44, '<')[:44]
    L2 = (_normalize_for_field(L2)).ljust(44, '<')[:44]
    return L1, L2

def validate_mrz_td3(raw_text: str):
    pair = _extract_td3_lines(raw_text)
    if not pair:
        return {"is_valid": False, "checks_passed": 0, "details": {"reason":"no_lines"}, "lines": ("","")}
    L1, L2 = pair
    if len(L2) != 44:
        return {"is_valid": False, "checks_passed": 0, "details": {"reason":"line2_len_fail"}, "lines": (L1, L2)}

    pn, pn_cd   = L2[0:9], L2[9]
    nat         = L2[10:13]
    dob, dob_cd = L2[13:19], L2[19]
    sex         = L2[20]
    exp, exp_cd = L2[21:27], L2[27]
    per, per_cd = L2[28:42], L2[42]
    comp_cd     = L2[43]

    def all_mrz_chars(s):
        return all(c.isdigit() or ('A'<=c<='Z') or c=='<' for c in s)

    line2_len_ok = (len(L2) == 44)
    nat_ok       = (len(nat) == 3 and nat.isalpha())
    dob_fmt_ok   = (len(dob) == 6 and dob.isdigit())
    exp_fmt_ok   = (len(exp) == 6 and exp.isdigit())
    sex_ok       = (sex in ("M","F","<"))
    per_fmt_ok   = (len(per) == 14 and all_mrz_chars(per))
    cds_fmt_ok = (all_mrz_chars(pn_cd) and all_mrz_chars(dob_cd)
                  and all_mrz_chars(exp_cd) and all_mrz_chars(per_cd)
                  and all_mrz_chars(comp_cd))
    schema_ok = (line2_len_ok and nat_ok and dob_fmt_ok and exp_fmt_ok
                 and sex_ok and per_fmt_ok and cds_fmt_ok)

    chk_pn  = (_compute_cd(pn)  == pn_cd)
    chk_dob = (_compute_cd(dob) == dob_cd)
    chk_exp = (_compute_cd(exp) == exp_cd)
    composite_data = L2[:10] + L2[13:20] + L2[21:28] + L2[28:43]
    chk_comp = (_compute_cd(composite_data) == comp_cd)

    checks_passed = sum([chk_pn, chk_dob, chk_exp, chk_comp])
    is_valid = (schema_ok and checks_passed >=1)
    return {
        "doc_type": "TD3",
        "is_valid": is_valid,
        "checks_passed": checks_passed,
        "details": {
            "schema_ok": schema_ok,
            "passport_number_cd": chk_pn,
            "date_of_birth_cd": chk_dob,
            "date_of_expiry_cd": chk_exp,
            "composite_cd": chk_comp,
            "nat_ok": nat_ok,
            "sex_ok": sex_ok,
            "dob_fmt_ok": dob_fmt_ok,
            "exp_fmt_ok": exp_fmt_ok,
            "per_fmt_ok": per_fmt_ok
        },
        "lines": (L1, L2)
    }

def guess_mrz_type_from_text(raw_text: str):
    lines = [ln.strip().upper() for ln in raw_text.splitlines() if ln.strip()]
    lines = [''.join(ch for ch in ln if ch.isalnum() or ch == '<') for ln in lines]
    if len(lines) >= 3 and all(28 <= len(ln) <= 32 for ln in lines[-3:]):
        return "TD1"
    if len(lines) >= 2 and all(40 <= len(ln) <= 46 for ln in lines[-2:]):
        return "TD3"
    return None

def validate_mrz_any(raw_text: str):
    t = guess_mrz_type_from_text(raw_text)
    if t == "TD3":
        v = validate_mrz_td3(raw_text)
        if v.get("is_valid") or v.get("lines", ("",""))[0]:
            v["doc_type"] = "TD3"; return v
        v2 = validate_mrz_td1(raw_text); v2["doc_type"] = "TD1"; return v2
    elif t == "TD1":
        v = validate_mrz_td1(raw_text)
        if v.get("is_valid") or v.get("lines", ("",""))[0]:
            v["doc_type"] = "TD1"; return v
        v2 = validate_mrz_td3(raw_text); v2["doc_type"] = "TD3"; return v2
    v1 = validate_mrz_td3(raw_text); v1["doc_type"] = "TD3"
    v2 = validate_mrz_td1(raw_text); v2["doc_type"] = "TD1"
    return v1 if v1.get("checks_passed",0) >= v2.get("checks_passed",0) else v2

def mrz_score_from_text(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    base = 0
    for ln in lines:
        if all((c.isalnum() or c == '<') for c in ln):
            L = len(ln); base += min(max(L, 30), 44)
    v = validate_mrz_any(text)
    cd_bonus = v["checks_passed"] * 30
    nat_bonus = 10 if v["details"].get("nat_ok") else 0
    head_bonus = 10 if v["details"].get("head_ok") else 0
    return base + cd_bonus + nat_bonus + head_bonus

def _normalize_date_separators(U: str) -> str:
    return re.sub(r'(?<!\d)([0-3]\d)\s+([01]\d)\s+(\d{4})(?!\d)', r'\1.\2.\3', U)

# =========================
# OCR (EasyOCR)
# =========================
def ocr_easy_text(img_bgr) -> tuple[list[str], str]:
    # ARTIK GLOBAL OKUYUCUYU KULLANIYORUZ!
    reader = get_easyocr_reader()
    lines = reader.readtext(img_bgr, detail=0, paragraph=True)
    lines = [ln.strip() for ln in lines if isinstance(ln, str) and ln.strip()]
    return lines, "\n".join(lines)

# --- RED-FLAG / STOPWORD SETİ ---
REDFLAG_WORDS = {
    "TÜRKİYE", "CUMHURİYETİ", "CUMHURIYETI", "TURKIYE",
    "SÜRÜCÜ", "BELGESİ", "SURUCU", "BELGESI",
    "DRIVING", "LICENCE", "LICENSE",
    "REPUBLIC", "OF", "TURKEY",
    "TR", "T.C", "TC", "PASSPORT", "KİMLİK", "KIMLIK"
}

HEADER_PATTERNS = [
    r"SÜRÜCÜ\s+BELGES[İI]",
    r"DRIVING\s+LICEN[SC]E",
    r"TÜRK[İI]YE\s+CUMHUR[İI]YET[İI]",
    r"\bREPUBLIC\s+OF\s+TURKEY\b",
    r"\bTR\b",
]

def _tr_upper(s: str) -> str:
    return (s or "").replace("i", "İ").replace("ı", "I").upper()

def _remove_headers(U: str) -> str:
    for pat in HEADER_PATTERNS:
        U = re.sub(pat, " ", U, flags=re.I)
    return U

def _remove_redflag_tokens(s: str) -> str:
    toks = _tokens(s)
    toks = [t for t in toks if _tr_upper(t) not in REDFLAG_WORDS]
    return " ".join(toks)

def _tokens(s: str) -> list[str]:
    toks = [t for t in re.split(r"[^\wÇĞİÖŞÜçğıöşü]+", s) if t]
    return [t for t in toks if _tr_upper(t) not in REDFLAG_WORDS and len(t) >= 2]

def _to_iso_tr_date(s: str) -> str | None:
    if not s: return None
    m = re.search(r'\b([0-3]?\d)[\s./-]([01]?\d)[\s./-](\d{4})\b', s)
    if not m: return None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        import datetime
        datetime.date(y, mth, d)
        return f"{y:04d}-{mth:02d}-{d:02d}"
    except Exception:
        return None

def is_valid_tckn(s: str) -> bool:
    if not re.fullmatch(r"\d{11}", s): return False
    if s[0] == "0": return False
    d = [int(c) for c in s]
    d10 = ((d[0]+d[2]+d[4]+d[6]+d[8]) * 7 - (d[1]+d[3]+d[5]+d[7])) % 10
    if d[9] != d10: return False
    d11 = sum(d[:10]) % 10
    return d[10] == d11

def extract_name_from_ocr_lines(lines: list[str]) -> tuple[str|None, str|None]:
    U0 = _tr_upper("\n".join(lines))
    U = _remove_headers(U0)
    m1 = re.search(r"\b1[.)]?\s*([A-ZÇĞİÖŞÜ]{2,})", U)
    m2 = re.search(r"\b2[.)]?\s*([A-ZÇĞİÖŞÜ ]{2,})", U)
    if m1 and m2:
        s = _tokens(m1.group(1))
        g = _tokens(m2.group(1))
        surname = s[-1] if s else None
        given   = " ".join(g) if g else None
        if surname and _tr_upper(surname) in REDFLAG_WORDS: surname = None
        if given and _tr_upper(given) in REDFLAG_WORDS: given = None
        if surname or given:
            return surname, given
    m3 = re.search(r"(.{0,80})\b3[.)]?\s*[0-3]\d[./-][01]\d[./-]\d{4}", U)
    if m3:
        left = _remove_headers(m3.group(1))
        toks = _tokens(left)
        window = toks[-5:] if len(toks) >= 5 else toks
        if len(window) >= 2:
            surname = window[-2]
            given   = " ".join(window[-1:])
            if _tr_upper(surname) in REDFLAG_WORDS and len(window) >= 3:
                surname = window[-3]
                given   = " ".join(window[-2:])
            return surname or None, (given or None)
    return None, None

def find_tckn_in_text(lines: list[str]) -> str|None:
    raw = " ".join(lines)
    for m in re.finditer(r"\b\d{11}\b", raw):
        cand = m.group(0)
        if is_valid_tckn(cand):
            return cand
    return None

def parse_turkish_driving_license(lines: list[str]) -> dict:
    U = _remove_headers(_tr_upper("\n".join(lines)))
    U = _remove_redflag_tokens(U)
    U = _normalize_date_separators(U)
    def g(pat):
        m = re.search(pat, U, flags=re.I)
        return m.group(1).strip() if m else None
    DATE_pat = r"([0-3]\d(?:[./\-\s])[01]\d(?:[./\-\s])\d{4})"
    dob_raw    = g(rf"\b3[.)]?\s*{DATE_pat}")
    issue_raw  = g(rf"\b4A[.)]?\s*{DATE_pat}")
    expiry_raw = g(rf"\b4B[.)]?\s*{DATE_pat}")
    place      = g(r"\b4C[.)]?\s*([A-Z0-9ÇĞİÖŞÜ \-]+?)(?:\s*\b4[BD][.)]?|\n|$)")
    serial     = g(r"\b4D[.)]?\s*([A-Z0-9]+)")
    number5    = g(r"\b5[.)]?\s*([A-Z0-9]+)")
    cats       = g(r"\b9[.)]?\s*([A-Z0-9ÇĞİÖŞÜ ]{2,})")
    surname, given = extract_name_from_ocr_lines(lines)
    tckn = find_tckn_in_text(lines)
    return {
        "doctype": "TR_DRIVING_LICENSE",
        "surname": surname,
        "given_name": given,
        "tckn": tckn,
        "birth_date_raw": dob_raw,
        "birth_date": _to_iso_tr_date(dob_raw or ""),
        "issue_date_raw": issue_raw,
        "issue_date": _to_iso_tr_date(issue_raw or ""),
        "expiry_date_raw": expiry_raw,
        "expiry_date": _to_iso_tr_date(expiry_raw or ""),
        "place_of_issue": (place or None),
        "serial_4d": serial,
        "number_5": number5,
        "categories_9": (cats or None),
        "raw_text": U,
    }

# =========================
# MRZ text extraction & field parsing
# =========================
def _extract_mrz_text(out_obj, text_fallback: str) -> str:
    if isinstance(out_obj, dict):
        t = (out_obj.get("raw_text") or out_obj.get("mrz_text") or "").strip()
        if t:
            return t
    if text_fallback:
        m = re.search(r'"mrz_text"\s*:\s*"([^"]+)"', text_fallback, flags=re.S)
        if m:
            t = m.group(1)
            try:
                t = bytes(t, "utf-8").decode("unicode_escape")
            except Exception:
                pass
            t = t.replace("\\n", "\n")
            return t.strip()
    return (text_fallback or "").strip()

def _yyMMdd_to_iso(s: str) -> Optional[str]:
    if not s or len(s) != 6 or not s.isdigit(): return None
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    now_yy = datetime.datetime.utcnow().year % 100
    century = 2000 if yy <= now_yy else 1900
    try:
        return f"{century+yy:04d}-{mm:02d}-{dd:02d}"
    except Exception:
        return None

def parse_fields_from_lines(doc_type: str, lines: Tuple[str, ...]) -> Dict:
    if doc_type == "TD1" and len(lines) >= 3:
        L1, L2, L3 = lines[0], lines[1], lines[2]
        fields = {
            "mrz_type": "TD1",
            "document_code": L1[0],
            "document_type_char": L1[1],
            "issuer_code": L1[2:5],
            "document_number": L1[5:14],
            "document_number_checkdigit": L1[14],
            "optional_data_1": L1[15:30].strip('<') or None,
            "birth_date_raw": L2[0:6],
            "birth_date": _yyMMdd_to_iso(L2[0:6]),
            "birth_date_checkdigit": L2[6],
            "sex": L2[7],
            "expiry_date_raw": L2[8:14],
            "expiry_date": _yyMMdd_to_iso(L2[8:14]),
            "expiry_date_checkdigit": L2[14],
            "nationality_code": L2[15:18],
            "optional_data_2": L2[18:29].strip('<') or None,
            "final_checkdigit": L2[29],
        }
        parts = L3.split("<<")
        surname = parts[0].replace("<"," ").strip()
        given = " ".join(p.replace("<"," ").strip() for p in parts[1:] if p)
        fields["surname"] = surname or None
        fields["given_name"] = given or None
        return fields
    if doc_type == "TD3" and len(lines) >= 2:
        L1, L2 = lines[0], lines[1]
        name_field = L1[5:].rstrip('<')
        if '<<' in name_field:
            surname_field, given_field = name_field.split('<<', 1)
        else:
            surname_field, given_field = name_field, ''
        surname = surname_field.replace('<', ' ').strip() or None
        given   = given_field.replace('<', ' ').strip() or None
        fields = {
            "mrz_type": "TD3",
            "document_code": L1[0],
            "issuer_code": L1[2:5],
            "passport_number": L2[0:9],
            "passport_number_checkdigit": L2[9],
            "nationality_code": L2[10:13],
            "birth_date_raw": L2[13:19],
            "birth_date": _yyMMdd_to_iso(L2[13:19]),
            "birth_date_checkdigit": L2[19],
            "sex": L2[20],
            "expiry_date_raw": L2[21:27],
            "expiry_date": _yyMMdd_to_iso(L2[21:27]),
            "expiry_date_checkdigit": L2[27],
            "personal_number": L2[28:42].strip('<') or None,
            "personal_number_checkdigit": L2[42],
            "final_checkdigit": L2[43],
            "surname": surname,
            "given_name": given,
        }
        return fields
    return {}

# =========================
# OCR runner (FastMRZ) — deskew yok
# =========================
def run_fastmrz(img, tesseract_path: Optional[str]=None, ignore_parse: bool=False):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if len(img.shape) == 2:
        img_to_save = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_to_save = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img_to_save = img
    h, w = img_to_save.shape[:2]
    if min(h, w) < 320:
        scale = 320.0 / float(min(h, w))
        img_to_save = cv2.resize(img_to_save, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    h, w = img_to_save.shape[:2]
    side = max(h, w)
    pt = (side - h) // 2; pb = side - h - pt
    pl = (side - w) // 2; pr = side - w - pl
    img_to_save = cv2.copyMakeBorder(img_to_save, pt, pb, pl, pr, borderType=cv2.BORDER_REPLICATE)
    tmp = f"temp_{uuid.uuid4().hex}.png"
    cv2.imwrite(tmp, img_to_save)
    try:
        fm = FastMRZ(tesseract_path=tesseract_path) if tesseract_path else FastMRZ()
        out = fm.get_details(tmp, ignore_parse=ignore_parse)
        if isinstance(out, dict):
            text = (out.get("raw_text") or out.get("mrz_text") or "").strip()
        else:
            text = str(out).strip()
        return out, text
    finally:
        try: os.remove(tmp)
        except: pass

def evaluate_variant_single(img, variant_name, tesseract_path: Optional[str]=None):
    # Performans takibi
    t0 = time.time()
    out, text = run_fastmrz(img, tesseract_path=tesseract_path, ignore_parse=False)
    
    text = _extract_mrz_text(out, text)
    s = mrz_score_from_text(text)
    v = validate_mrz_any(text)
    Ls = v["lines"]
    doc_type = v.get("doc_type", "unknown")
    if isinstance(out, dict) and out.get("mrz_type"):
        doc_type = out["mrz_type"]
    
    elapsed = time.time() - t0
    # Log: Hangi varyantın ne kadar sürdüğünü görmek için
    # log.info(f"Varyant: {variant_name} Bitti. Süre: {elapsed:.3f}s")
    
    row = {
        "variant": variant_name,
        "best_mode": "parsed",
        "score": s,
        "checks_passed": v["checks_passed"],
        "is_valid": v["is_valid"],
        "mrz_line1": Ls[0] if len(Ls) >= 1 else "",
        "mrz_line2": Ls[1] if len(Ls) >= 2 else "",
        "doc_type": doc_type,
        "text_preview": text[:600],
        "raw_text": text
    }
    if len(Ls) > 2:
        row["mrz_line3"] = Ls[2]
    return row, out

# =========================
# Orchestrator
# =========================

def paddle_front_ocr_and_compare(best_img: np.ndarray, mrz_fields: Dict) -> Dict:
    """
    best_img: MRZ için seçilen en iyi varyantın görüntüsü (BGR numpy)
    mrz_fields: parse_fields_from_lines ile elde edilen MRZ alanları

    Dönen:
      {
        "ok": bool,
        "overall_score": 0-100,
        "visual_fields": {...},
        "per_field": {...},
        "raw_text": "..."
      }
    """
    result: Dict[str, Any] = {
        "ok": False,
        "overall_score": 0.0,
        "visual_fields": {},
        "per_field": {},
        "raw_text": ""
    }

    if best_img is None:
        result["reason"] = "no_best_image"
        return result

    # --------- PaddleOCR init (global cache) ----------
    global GLOBAL_PADDLE_OCR
    if GLOBAL_PADDLE_OCR is None:
        log.info("⚡ PaddleOCR (front) modeli RAM'e yükleniyor...")
        GLOBAL_PADDLE_OCR = PaddleOCR(
            det_model_dir="models/PP-OCRv5_server_det_infer",
            rec_model_dir="models/PP-OCRv5_server_rec_infer",
            use_angle_cls=True,
            lang="en",
            #use_gpu=False,
            det_limit_side_len=2048,
            det_limit_type='max',
            det_db_thresh=0.25,
            det_db_box_thresh=0.7,
        )

    ocr = GLOBAL_PADDLE_OCR

    # --------- Görüntüyü OCR'a ver ----------
    try:
        img_rgb = cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB)
    except Exception:
        img_rgb = best_img

    ocr_res = ocr.ocr(img_rgb, cls=True)
    lines: List[str] = []
    if ocr_res and len(ocr_res) > 0:
        for line in ocr_res[0]:
            try:
                text = line[1][0]
                score = float(line[1][1])
            except Exception:
                text, score = str(line), 1.0
            text = text.strip()
            if not text:
                continue
            # çok düşük skorları at
            if score < 0.40:
                continue
            lines.append(text)

    raw_text = "\n".join(lines)
    result["raw_text"] = raw_text

    # --------- Görsel taraftan alan çıkarımı (MRZ'e göre akıllı) ----------

    # Tüm token'ları çıkar
    tokens: list[str] = []
    for ln in lines:
        for tok in re.split(r"[^\w]+", ln):
            tok = tok.strip()
            if not tok:
                continue
            tokens.append(tok.upper())

    # Tekrarlananları at, sıra korunsun
    uniq_tokens: list[str] = list(dict.fromkeys(tokens))

    # Yardımcı: isim normalize
    def _norm_name(s: str | None) -> str:
        if not s:
            return ""
        s = s.upper()
        s = re.sub(r'[^A-Z0-9ÇĞİÖŞÜ]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # MRZ alanları
    mrz_surname = _norm_name(mrz_fields.get("surname"))
    mrz_given   = _norm_name(mrz_fields.get("given_name"))
    mrz_doc_raw = mrz_fields.get("document_number") or mrz_fields.get("passport_number") or ""
    mrz_doc_norm = re.sub(r'[^A-Z0-9]', '', (mrz_doc_raw or "").upper())

    # --- Doküman numarası: rakam içeren 6–12 karakterlik tokenlar arasından MRZ'e en benzeyeni seç ---
    visual_doc_no = None
    best_doc_sim = 0.0
    cand_doc_tokens = [
        t for t in uniq_tokens
        if 6 <= len(t) <= 12 and re.search(r'\d', t)
    ]
    for t in cand_doc_tokens:
        sim = levenshtein_ratio(mrz_doc_norm, t)
        if sim > best_doc_sim:
            best_doc_sim = sim
            visual_doc_no = t
    if best_doc_sim < 0.60:
        # çok düşükse güvenme
        visual_doc_no = None

    # --- İsim / soyisim: rakamsız tokenlar arasından MRZ değerine en yakın olanı bul ---
    name_tokens = [t for t in uniq_tokens if not re.search(r'\d', t) and len(t) >= 2]

    visual_surname = None
    best_surname_sim = 0.0
    if mrz_surname:
        for t in name_tokens:
            sim = levenshtein_ratio(mrz_surname, t)
            if sim > best_surname_sim:
                best_surname_sim = sim
                visual_surname = t
        if best_surname_sim < 0.60:
            visual_surname = None

    visual_given = None
    best_given_sim = 0.0
    if mrz_given:
        for t in name_tokens:
            sim = levenshtein_ratio(mrz_given, t)
            if sim > best_given_sim:
                best_given_sim = sim
                visual_given = t
        if best_given_sim < 0.60:
            visual_given = None

    # --- Doğum tarihi dd.mm.yyyy / dd-mm-yyyy vs. (pasaportta olmayabilir) ---
    # --- Doğum tarihi: MRZ doğum tarihi ile eşleşeni bul ---
    visual_dob_iso = None
    mrz_dob_iso = mrz_fields.get("birth_date")

    if mrz_dob_iso:
        # 1) MRZ tarihinden ddmmyy string üret (ör: 1999-11-06 -> "061199")
        ddmmyy = None
        try:
            y_mrz, m_mrz, d_mrz = map(int, mrz_dob_iso.split("-"))
            ddmmyy = f"{d_mrz:02d}{m_mrz:02d}{y_mrz % 100:02d}"
        except Exception:
            ddmmyy = None

        # 1a) ddmmyy token'ı doğrudan geçtiyse (BEL örneği: "061199")
        if ddmmyy:
            for t in uniq_tokens:
                if t == ddmmyy:
                    visual_dob_iso = mrz_dob_iso
                    break

        # 2) dd[sep]mm[sep]yyyy / dd[sep]mm[sep]yy formatlarını tara
        if visual_dob_iso is None:
            for m_dob in re.finditer(r'\b([0-3]?\d)[\s./-]([01]?\d)[\s./-](\d{2,4})\b', raw_text):
                d_str, mo_str, y_str = m_dob.group(1), m_dob.group(2), m_dob.group(3)
                try:
                    d = int(d_str)
                    mo = int(mo_str)
                    y = int(y_str)
                except Exception:
                    continue

                # 2 haneli yılı 19xx / 20xx'e map et (MRZ'deki mantığa benzer)
                if y < 100:
                    yy = y
                    now_yy = datetime.datetime.utcnow().year % 100
                    century = 2000 if yy <= now_yy else 1900
                    y = century + yy

                try:
                    datetime.date(y, mo, d)
                except Exception:
                    continue

                iso = f"{y:04d}-{mo:02d}-{d:02d}"
                if iso == mrz_dob_iso:
                    visual_dob_iso = iso
                    break

        # 3) ddMONyyyy (01JAN1975, 01 JAN 1975, O1JAN 1975 vb.)
        if visual_dob_iso is None:
            month_map = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                "JUL": 7, "AUG": 8, "SEP": 9, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12
            }
            upper_text = raw_text.upper()
            # O1JAN 1975 gibi şeyleri yakalamak için [0-3O]?
            pattern = r'\b([0-3O]?\d)\s*([A-Z]{3,4})\s*(\d{4})\b'
            for m_dob in re.finditer(pattern, upper_text):
                day_s, mon_s, year_s = m_dob.group(1), m_dob.group(2), m_dob.group(3)
                # OCR: "O1" -> "01" düzelt
                day_s = day_s.replace("O", "0")
                try:
                    d = int(day_s)
                    y = int(year_s)
                except Exception:
                    continue

                mon_s3 = mon_s[:3]
                mo = month_map.get(mon_s3)
                if not mo:
                    continue

                try:
                    datetime.date(y, mo, d)
                except Exception:
                    continue

                iso = f"{y:04d}-{mo:02d}-{d:02d}"
                if iso == mrz_dob_iso:
                    visual_dob_iso = iso
                    break

    # (visual_dob_iso bulunduysa biraz aşağıda visual_fields içine koyacağız)


    # --- Cinsiyet ---
    visual_sex = None
    m_sex = re.search(r'\b(MALE|FEMALE|M|F)\b', raw_text, flags=re.I)
    if m_sex:
        s = m_sex.group(1).upper()
        if s in ("M", "MALE"):
            visual_sex = "M"
        elif s in ("F", "FEMALE"):
            visual_sex = "F"

    # --- Uyruk (3 harfli kod) ---
    visual_nat = None
    m_nat = re.search(r'\b([A-Z]{3})\b', raw_text.upper())
    if m_nat:
        visual_nat = m_nat.group(1)

    visual_fields = {
        "surname": visual_surname,
        "given_name": visual_given,
        "document_number": visual_doc_no,
        "birth_date": visual_dob_iso,
        "sex": visual_sex,
        "nationality": visual_nat,
    }
    result["visual_fields"] = visual_fields


    # --------- MRZ vs Görsel alan karşılaştırma ----------
    def norm_name(s: str | None) -> str:
        if not s:
            return ""
        s = s.upper()
        s = re.sub(r'[^A-Z0-9ÇĞİÖŞÜ]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    per_field: Dict[str, Any] = {}
    matched = 0
    possible = 0

    # Surname
    mrz_surname = norm_name(mrz_fields.get("surname"))
    vis_surname_n = norm_name(visual_fields.get("surname"))
    if mrz_surname or vis_surname_n:
        possible += 1
        if mrz_surname and vis_surname_n:
            sim = levenshtein_ratio(mrz_surname, vis_surname_n)
            match = sim >= 0.75
            if match:
                matched += 1
            per_field["surname"] = {
                "mrz": mrz_surname,
                "visual": vis_surname_n,
                "similarity": round(sim, 3),
                "match": match,
            }
        else:
            per_field["surname"] = {
                "mrz": mrz_surname,
                "visual": vis_surname_n,
                "similarity": None,
                "match": False,
            }

    # Given name
    mrz_given = norm_name(mrz_fields.get("given_name"))
    vis_given_n = norm_name(visual_fields.get("given_name"))
    if mrz_given or vis_given_n:
        possible += 1
        if mrz_given and vis_given_n:
            sim = levenshtein_ratio(mrz_given, vis_given_n)
            match = sim >= 0.70
            if match:
                matched += 1
            per_field["given_name"] = {
                "mrz": mrz_given,
                "visual": vis_given_n,
                "similarity": round(sim, 3),
                "match": match,
            }
        else:
            per_field["given_name"] = {
                "mrz": mrz_given,
                "visual": vis_given_n,
                "similarity": None,
                "match": False,
            }

    # Document number
    mrz_doc = mrz_fields.get("document_number") or mrz_fields.get("passport_number") or ""
    mrz_doc_n = re.sub(r'[^A-Z0-9]', '', (mrz_doc or "").upper())
    vis_doc_n = re.sub(r'[^A-Z0-9]', '', (visual_fields.get("document_number") or "").upper())
    if mrz_doc_n or vis_doc_n:
        possible += 1
        if mrz_doc_n and vis_doc_n:
            sim = 1.0 if mrz_doc_n == vis_doc_n else levenshtein_ratio(mrz_doc_n, vis_doc_n)
            match = sim >= 0.90
            if match:
                matched += 1
            per_field["document_number"] = {
                "mrz": mrz_doc_n,
                "visual": vis_doc_n,
                "similarity": round(sim, 3),
                "match": match,
            }
        else:
            per_field["document_number"] = {
                "mrz": mrz_doc_n,
                "visual": vis_doc_n,
                "similarity": None,
                "match": False,
            }

    # Birth date (iso)
    mrz_dob = mrz_fields.get("birth_date")
    vis_dob = visual_fields.get("birth_date")
    if mrz_dob or vis_dob:
        possible += 1
        match = (bool(mrz_dob) and bool(vis_dob) and str(mrz_dob) == str(vis_dob))
        if match:
            matched += 1
        per_field["birth_date"] = {
            "mrz": mrz_dob,
            "visual": vis_dob,
            "similarity": 1.0 if match else 0.0,
            "match": match,
        }

    # Sex
    mrz_sex = (mrz_fields.get("sex") or "").upper()[:1]
    vis_sex = (visual_fields.get("sex") or "").upper()[:1]
    if mrz_sex or vis_sex:
        possible += 1
        match = bool(mrz_sex and vis_sex and mrz_sex == vis_sex)
        if match:
            matched += 1
        per_field["sex"] = {
            "mrz": mrz_sex,
            "visual": vis_sex,
            "similarity": 1.0 if match else 0.0,
            "match": match,
        }

    # Nationality
    mrz_nat = (mrz_fields.get("nationality_code") or mrz_fields.get("issuer_code") or "").upper()
    vis_nat = (visual_fields.get("nationality") or "").upper()
    if mrz_nat or vis_nat:
        possible += 1
        match = bool(mrz_nat and vis_nat and mrz_nat == vis_nat)
        if match:
            matched += 1
        per_field["nationality"] = {
            "mrz": mrz_nat,
            "visual": vis_nat,
            "similarity": 1.0 if match else 0.0,
            "match": match,
        }

    # Genel skor
    if possible > 0:
        overall = (matched / possible) * 100.0
    else:
        overall = 0.0

    result["overall_score"] = round(overall, 2)
    result["ok"] = overall >= 70.0  # eşiği istersen değiştir
    result["per_field"] = per_field

    return result

def process_document(
    file_bytes: bytes,
    *,
    tesseract_path: Optional[str] = TESSERACT_PATH,
    workers: Optional[int] = None,
) -> Tuple[Dict, Optional[np.ndarray]]:
    
    if not file_bytes or len(file_bytes) < 10:
        raise ProcessingError("EMPTY_FILE", "Dosya boş görünüyor")

    # 1. Görüntüyü Oku
    original_color = imread_color_bytes(file_bytes)
    
    # 2. MRZ işlemleri için griye çevir ve küçült
    processing_img = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)
    processing_img = resize_max(processing_img, MAX_SIDE)
    
    variants = pipelines(processing_img)
    rows: List[Dict] = []
    imgs_by_index: Dict[int, np.ndarray] = {}

    # --- DÜZELTME: MacOS için Multiprocessing yerine Döngü Kullanımı ---
    # ProcessPoolExecutor kaldırıldı, güvenli döngü eklendi.
    log.info("🐢 İşlemler sıralı (serial) modda çalıştırılıyor...")
    
    for i, (name, im) in enumerate(variants, start=1):
        imgs_by_index[i] = im
        try:
            # İşlemi direkt çağırıyoruz (Process açmadan)
            eval_row, _parsed = evaluate_variant_single(im, name, tesseract_path)
            eval_row["index"] = i
            rows.append(eval_row)
        except Exception as e:
            log.warning(f"Varyant {name} hata: {e}")

    # Hiç MRZ satırı oluşmadıysa (Fallback: OCR)
    if not rows:
        color = imread_color_bytes(file_bytes)
        ocr_lines, _ocr_text = ocr_easy_text(color)
        
        # Sadece ehliyet fonksiyonunu çağırmıyoruz, çünkü pasaport olabilir
        # fields_ocr = parse_turkish_driving_license(ocr_lines) -> BU KALDIRILDI
        
        best = {
            "variant": "ocr",
            "doc_type": "NONE",
            "is_valid": False,
            "checks_passed": 0,
            "mrz": {},
            "ocr": {"lines": ocr_lines},
            "fields": {}, # Yanlış veri üretmemesi için boş
            "debug": {"reason": "no_mrz_variants"}
        }
        return {"best": best, "variants": []}, None

    for r in rows:
        r["final_score"] = r["checks_passed"] * 100 + r["score"]

    best_row = max(rows, key=lambda x: (x.get("checks_passed", 0), x.get("final_score", 0)))

    # MRZ Geçersizse (Fallback: OCR)
    if best_row.get("checks_passed", 0) == 0 or not best_row.get("is_valid", False):
        log.info("MRZ doğrulama başarısız. Fallback: EasyOCR devreye giriyor...")
        color = imread_color_bytes(file_bytes)
        ocr_lines, _ocr_text = ocr_easy_text(color)
        
        best = {
            "variant": "ocr",
            "doc_type": "NONE",
            "is_valid": False,
            "checks_passed": 0,
            "mrz": {},
            "ocr": {"lines": ocr_lines},
            "fields": {},
            "debug": {"reason": "mrz_invalid"}
        }
        return {"best": best, "variants": rows}, None

    # MRZ GEÇERLİ
    best_idx = best_row["index"]
    best_img = imgs_by_index.get(best_idx, None)
    doc_type = best_row.get("doc_type") or "unknown"

    lines = (
        best_row.get("mrz_line1", ""),
        best_row.get("mrz_line2", ""),
        best_row.get("mrz_line3", ""),
    )
    fields = parse_fields_from_lines(doc_type, lines) if best_row.get("is_valid") else {}

    result_best = {
        "variant": best_row["variant"],
        "doc_type": doc_type,
        "is_valid": bool(best_row["is_valid"]),
        "checks_passed": int(best_row["checks_passed"]),
        "mrz": {
            "line1": lines[0],
            "line2": lines[1],
            **({"line3": lines[2]} if lines[2] else {})
        },
        "fields": fields,
        "debug": {
            "index": best_idx,
            "score": best_row.get("score", 0),
            "final_score": best_row.get("final_score", 0),
        },
    }

    # MRZ + PaddleOCR görsel karşılaştırma
    try:
        if best_img is not None:
             compare_res = paddle_front_ocr_and_compare(best_img, fields)
             result_best["visual_compare"] = compare_res
    except Exception as e:
        log.warning(f"PaddleOCR compare hata: {e}")

    return {"best": result_best, "variants": rows}, best_img


# =========================
# FastAPI app
# =========================
app = FastAPI(title="MRZ Scanner API (Optimized)", version="3.2.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/mrz/scan")
async def mrz_scan(
    file: UploadFile = File(...),
    include_image_base64: bool = Query(False, description="En iyi varyant görselini data-url olarak döndür"),
    include_variants: bool = Query(False, description="Debug için tüm varyantları ekle"),
    workers: int | None = Query(None, description="Paralel süreç sayısı"),
):
    start_time = time.time()
    try:
        process_start = time.time()
        data = await file.read()
        
        # RAM bol olduğu için workers kısıtlamasına gerek yok, 
        # ama EasyOCR yüklemesi artık optimize.
        result, best_img = process_document(
            data,
            tesseract_path=TESSERACT_PATH,
            workers=workers,
        )
        process_time = time.time() - process_start

        response_start = time.time()
        resp = {"result": {"source":"MRZ", "best": result["best"]}}
        if include_variants:
            resp["result"]["variants"] = result["variants"]
        if include_image_base64 and best_img is not None:
            b64 = base64.b64encode(encode_png(best_img)).decode("ascii")
            resp["best_image_data_url"] = f"data:image/png;base64,{b64}"
        
        response_time = time.time() - response_start
        total_time = time.time() - start_time

        log.info(f"✅ MRZ Scan success - Total: {total_time:.2f}s (Proc: {process_time:.2f}s)")

        return JSONResponse(resp, headers={"Cache-Control":"no-store","Pragma":"no-cache"})
    except ProcessingError as e:
        error_time = time.time() - start_time
        log.error(f"❌ MRZ scan failed (ProcessingError) in {error_time:.2f}s: {e.code} - {e.message}")
        raise HTTPException(status_code=422, detail={"code": e.code, "message": e.message})
    except Exception as e:
        error_time = time.time() - start_time
        log.error(f"❌ MRZ scan failed (Unexpected) in {error_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail={"code":"INTERNAL_ERROR","message":str(e)})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Test görseli (JPG/PNG/PDF)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()

    if not args.image:
        print("CLI için --image verin. API için: uvicorn mrzLnx:app --host 0.0.0.0 --port 8001")
        raise SystemExit(0)

    with open(args.image, "rb") as f:
        data = f.read()
    result, _ = process_document(data, tesseract_path=TESSERACT_PATH, workers=args.workers)
    print(json.dumps({"source":"MRZ", "best": result["best"]}, ensure_ascii=False, indent=2))