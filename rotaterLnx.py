from __future__ import annotations
# app.py
import os
import re
import io
import cv2
import fitz  # PyMuPDF
import base64
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from ultralytics import YOLO
import ocrmypdf
import shutil
import pytesseract


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'temp_files'
OUTPUT_FOLDER = 'output_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Poppler path: Linux/Server'da genelde PATH'te olur; macOS Homebrew: /opt/homebrew/opt/poppler/bin
# Windows'ta poppler kurulu klasörün "bin" yolunu buraya verin.
POPPLER_PATH = r'/usr/bin'  # gerekirse değiştirin
MODEL_PATH = 'yolov8s.pt'         # kendi model dosyanız
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# -----------------------------------------------------------------------------
# YOLO Model Load
# -----------------------------------------------------------------------------
yolo_model = None
try:
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
        logger.info(f"YOLOv8 imza tespit modeli yerel yoldan yüklendi: {MODEL_PATH}")
    else:
        logger.error(f"Belirtilen YOLOv8 model dosyası bulunamadı: {MODEL_PATH}. YOLO tabanlı kontrol devre dışı.")
except Exception as e:
    logger.error(f"YOLOv8 modeli yüklenirken hata oluştu: {e}")

# -----------------------------------------------------------------------------
# PDF İşleme (OCR, döndürme vb.)
# -----------------------------------------------------------------------------
def process_and_correct_pdf(input_pdf_path: str, output_pdf_path: str) -> bool:
    """
    ocrmypdf ile belgeyi işler. force_ocr=False: seçilebilir metin varsa OCR'e zorlamaz.
    """
    try:
        if os.path.exists(output_pdf_path):
            os.remove(output_pdf_path)
            logger.info(f"Mevcut çıkış dosyası silindi: {output_pdf_path}")

        ocrmypdf.ocr(
            input_pdf_path,
            output_pdf_path,
            force_ocr=True, #zorunlu olmadıkça açılmaması gereken bir parametre programı çok yavaşlatıyor. 
            rotate_pages=True,
            rotate_pages_threshold=1.0,  # isterseniz açabilirsiniz
            output_type='pdf',
            language= 'tur+eng',
            verbose=2  # Türkçe OCR için
        )
        logger.info(f"ocrmypdf: Belge başarıyla işlendi ve kaydedildi: {output_pdf_path}")
        return True
    except Exception as e:
        logger.error(f"ocrmypdf: Belge işlenirken bir hata oluştu: {e}")
        return False

# -----------------------------------------------------------------------------
# Reference No Extraction 
# -----------------------------------------------------------------------------


def _pdf_text(doc: fitz.Document) -> str:
    parts = []
    for p in doc:
        parts.append(p.get_text("text") or "")
    return "\n".join(parts).strip()

def _extract_11_digit_from_text(text: str) -> str | None:
    if not text:
        return None

    # Etiket + 11 hane (yaygın varyasyonlar)
    label_patterns = [
        r"SENDERMONEY\s+REF\.?\s*NO\s*/\s*TRX\s*REF\.?\s*NO\s*:\s*([0-9]{11})",
        r"TRX\s*REF\.?\s*NO\s*:\s*([0-9]{11})",
    ]
    for pat in label_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    # Etiket bulunursa sağ taraftaki kısa pencerede ara (satır atlamaları vs. için)
    generic_labels = [
        r"SENDERMONEY\s+REF\.?\s*NO\s*/\s*TRX\s*REF\.?\s*NO",
        r"TRX\s*REF\.?\s*NO",
    ]
    for lab in generic_labels:
        for m in re.finditer(lab, text, flags=re.IGNORECASE):
            window = text[m.end(): m.end() + 120]
            mnum = re.search(r"\b([0-9]{11})\b", window)
            if mnum:
                return mnum.group(1)

    return None

def _try_extract_reference_from_pdf(pdf_path: str) -> str | None:
    """PyMuPDF çıktısından Reference No (11 hane) arar."""
    try:
        with fitz.open(pdf_path) as doc:
            full_text = _pdf_text(doc)
        ref = _extract_11_digit_from_text(full_text)
        if ref:
            logger.info(f"Referans numarası bulundu (metin): {ref}")
        return ref
    except Exception as e:
        logger.exception(f"PDF metin çıkarımında hata: {e}")
        return None
    
# -----------------------------------------------------------------------------
# Basit Dekont Sayısı Tespiti (Regex ile)
# -----------------------------------------------------------------------------
PATTERNS_RECEIPT = [
    r"SENDERMONEY\s+REF\.?\s*NO\s*/\s*TRX\s*REF\.?\s*NO",
    r"Dekont-Ödeme\s+Kuruluşu",
    r"KURUM\s+REF\.?\s*NO\s*/\s*PARTNER\s*REF\.?\s*NO"
]

def detect_receipt_count(pdf_path: str) -> int:
    """
    PDF'teki dekont sayısını (1 veya 2) döndürür.
    Basit: belirli başlık/desenlerin toplam tekrarını sayar.
    Belirsizse 2'ye varsayar (temkinli yaklaşım).
    """
    max_count = 0
    try:
        with fitz.open(pdf_path) as doc:
            joined = []
            for page in doc:
                joined.append(page.get_text("text") or "")
            full_text = "\n".join(joined)
            for pattern in PATTERNS_RECEIPT:
                m = re.findall(pattern, full_text, flags=re.IGNORECASE)
                max_count = max(max_count, len(m))
    except Exception as e:
        logger.error(f"PDF okunurken hata (detect_receipt_count): {e}")
    return max_count

def _ocr_and_retry_reference(pdf_path: str) -> str | None:
    """
    OCR uygulanmış geçici PDF üretip tekrar dener.
    process_and_correct_pdf fonksiyonunu kullanır.
    """
    try:
        tmp_out = os.path.join(app.config['OUTPUT_FOLDER'], f"ocr_{os.urandom(6).hex()}.pdf")
        ok = process_and_correct_pdf(pdf_path, tmp_out)
        if ok and os.path.exists(tmp_out):
            try:
                ref = _try_extract_reference_from_pdf(tmp_out)
                return ref
            finally:
                try:
                    os.remove(tmp_out)
                except Exception:
                    pass
        return None
    except Exception as e:
        logger.exception(f"OCR fallback sırasında hata: {e}")
        return None

def detect_correction_angle(img_bgr=None, pdf_path=None, dpi=400, short_side=768):
    """
    Tek sayfa belge için DÜZELTME açısı (delta) -> {0, 90, 180, 270}
    Önce Tesseract OSD dener, başarısızsa Sobel+yoğunluk fallback.
    """
    if img_bgr is None and pdf_path is None:
        raise ValueError("detect_correction_angle: img_bgr veya pdf_path parametresinden en az biri verilmelidir.")

    # 1) Girdi görüntü (yoksa PDF'ten küçük raster)
    if img_bgr is None:
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR) if pix.n == 4 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        doc.close()

    h, w = img_bgr.shape[:2]
    scale = short_side / float(min(h, w))
    img_small = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img_bgr
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    # 1.a) OSD dene (başarırsa direkt doğru delta budur)
    try:
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)  # --psm 0
        angle_osd = int(osd.get("rotate", 0)) % 360
        return angle_osd
    except Exception as e:
        logger.debug(f"OSD orientation failed, fallback to Sobel: {e}")

    # 2) Fallback: Sobel + adaptif eşik (mevcut mantığın iyileştirilmiş hali)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sum_gx = float(np.sum(np.abs(gx)))
    sum_gy = float(np.sum(np.abs(gy)))
    horizontal_like = (sum_gx >= sum_gy)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    H, W = thr.shape

    if horizontal_like:
        top = np.sum(thr[: H//4, :]) / 255.0
        bot = np.sum(thr[3*H//4 :, :]) / 255.0
        current = 0 if top >= bot else 180
    else:
        left  = np.sum(thr[:, : W//4]) / 255.0
        right = np.sum(thr[:, 3*W//4 :]) / 255.0
        current = 90 if right >= left else 270

    delta = int((360 - current) % 360)
    return delta



def rotate_rgb_90s(arr, angle):
    """NumPy RGB görüntüyü 0/90/180/270 derece döndür (saat yönünde)."""
    if angle % 360 == 0:
        return arr
    if angle % 360 == 90:
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    if angle % 360 == 180:
        return cv2.rotate(arr, cv2.ROTATE_180)
    if angle % 360 == 270:
        return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return arr

def find_reference_num(input_pdf_path: str, use_ocr_fallback: bool = True) -> str | None:
    """
    'Reference No / Ref. No / REF.NO / TRX REF.NO / SENDERMONEY REF.NO/TRX REF.NO'
    etiketlerinin yanındaki 11 haneli referansı döndürür. Bulamazsa None.
    """
    # 1) Doğrudan metinden dene
    ref = _try_extract_reference_from_pdf(input_pdf_path)
    if ref:
        return ref

    # 2) Gerekirse OCR ile dene
    if use_ocr_fallback:
        ref = _ocr_and_retry_reference(input_pdf_path)
        if ref:
            logger.info(f"Referans numarası bulundu (OCR): {ref}")
            return ref

    logger.warning("Referans numarası bulunamadı.")
    return None


"""def detect_signatures():

    yolo_model    
    image_bgr_or_pil,
    # En çok etki edenler:
    conf=0.20,          # default ~0.25; kaçanları yakalamak için düşür
    iou=0.45,           # NMS eşik; yakın kutuları birleştirmeyi etkiler
    imgsz=1280,         # giriş boyutu; 1280 (veya 1536) küçük imzaları yakalamada iyi
    max_det=300,        # bir sayfada çok obje varsa artır
    classes=None,       # sadece 'signature' sınıfının id'si ise örn. [0] ver
    agnostic_nms=True,  # sınıf bağımsız NMS; yakın kutuların elenmesini azaltır

    # Faydalı ekler:
    augment=True,       # TTA (scale/flip) ile recall artar, biraz yavaşlatır
    half=False,         # GPU’da bellek kısar; T4/Ampere’de iyi; CPU’da False kalsın
    device=None,        # "0" (ilk GPU) / "cpu"; None = otomatik
    verbose=False"""


def is_within_customer_signature_area(x_center, y_center, image_shape):
    """
    Heuristik ROI: sayfanın alt-sağ kuşağı (müşteri imzası için).
    Gerekirse oranları ayarlayabilirsiniz.
    """
    h, w = image_shape[:2]
    roi_top = int(h * 0.70)
    roi_bottom = int(h * 0.83)
    roi_left = int(w * 0.60)
    roi_right = int(w * 0.90)
    return roi_left <= x_center <= roi_right and roi_top <= y_center <= roi_bottom

def check_signature_logic_yolo(image_np, debug_image_path=None, receipt_count=2):
    if yolo_model is None:
        logger.error("YOLOv8 modeli hazır değil.")
        return False, "YOLOv8 modeli hazır değil.", None
    if image_np is None:
        return False, "Görüntü boş.", None

    img_bgr = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)
    results = yolo_model(image_np, verbose=False)
    detected_signatures = []

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls.item())
            label = yolo_model.names.get(class_id, f"Sınıf {class_id}")
            confidence = float(box.conf.item()) # 0.0 - 1.0 aralığında bir değer, imzadan ne kadar emin olduğunu gösterir. 
            coords = box.xyxy.int().tolist()[0]
            x1, y1, x2, y2 = coords
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            if label == 'signature' and confidence > 0.3:
                detected_signatures.append({
                    "x_center": x_center,
                    "y_center": y_center,
                    "confidence": confidence,
                    "coords": coords
                })
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, f"{confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    customer_signature_found = False
    message = "YOLOv8 kontrolü tamamlandı, müşteri imzası bulunamadı."


    if receipt_count ==1:
        if detected_signatures:
            # Sağdan sola sırala
            detected_signatures.sort(key=lambda s: s['x_center'], reverse=True)

            h, w = img_bgr.shape[:2]
            mid_x = w / 2.0
            tol = int(0.03 * w)  # orta çizgi için %3 tolerans bandı

            # Sağ yarıda olan ilk imzayı bul (orta çizginin tol kadar sağında)
            right_side = [s for s in detected_signatures if s['x_center'] >= (mid_x + tol)]

            if right_side:
                rightmost = right_side[0]
                customer_signature_found = True
                x1, y1, x2, y2 = rightmost['coords']
                message = (f"{len(detected_signatures)} imza tespit edildi. En sağdaki müşteri imzası "
                        f"olarak kabul edildi (Güven: {rightmost['confidence']:.2f}).")
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img_bgr, "CUSTOMER", (x1, max(10, y1 - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            else:
                # Sağ yarıda imza yok → müşteri imzası sayma
                customer_signature_found = False
                message = (f"{len(detected_signatures)} imza tespit edildi fakat hiçbiri sayfanın "
                        f"sağ yarısında değil. Müşteri imzası bulunamadı.")
                # (İstersen burada en sağdaki kutuyu DEBUG amaçlı mor çiz ve 'RIGHT-NOT-IN-RIGHT-HALF' yaz)
                x1, y1, x2, y2 = detected_signatures[0]['coords']
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (180, 0, 180), 2)
                cv2.putText(img_bgr, "Operator", (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 0, 180), 2)
            
        #debug için görseli kaydet
        if debug_image_path: 
            try:
                cv2.imwrite(debug_image_path, img_bgr)
            except Exception as e:
                logger.warning(f"YOLO debug görseli kaydedilemedi: {e}")
        return customer_signature_found, message, img_bgr
    

    if detected_signatures:
        num_sigs = len(detected_signatures)

        if num_sigs in [1, 2]:
            roi_matched = None
            for sig in detected_signatures:
                x_c, y_c = sig['x_center'], sig['y_center']
                if is_within_customer_signature_area(x_c, y_c, image_np.shape):
                    roi_matched = sig
                    break
            if roi_matched:
                customer_signature_found = True
                message = f"{num_sigs} imza tespit edildi, biri müşteri alanında (Güven: {roi_matched['confidence']:.2f})."
                x1, y1, x2, y2 = roi_matched["coords"]
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img_bgr, "CUSTOMER", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            else:
                message = f"{num_sigs} imza bulundu ama hiçbiri müşteri alanında değil."
        else:
            detected_signatures.sort(key=lambda s: s['x_center'], reverse=True)
            rightmost = detected_signatures[0]
            customer_signature_found = True
            message = (f"{num_sigs} imza tespit edildi. En sağdaki müşteri imzası "
                       f"olarak kabul edildi (Güven: {rightmost['confidence']:.2f}).")
            x1, y1, x2, y2 = rightmost['coords']
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img_bgr, "CUSTOMER", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    if debug_image_path and img_bgr is not None:
        try:
            cv2.imwrite(debug_image_path, img_bgr)
        except Exception as e:
            logger.warning(f"YOLO debug görseli kaydedilemedi: {e}")

    return customer_signature_found, message, img_bgr

@app.route('/check_document_signature', methods=['POST'])
def check_document_signature():
    """
    PDF dosyasını multipart/form-data ile 'file' alanından alır.
    - Reference No (11 hane) çıkarır (OCR fallback dahil).
    - Son sayfayı görüntüye çevirir ve YOLO imza kontrolünü çalıştırır.
    - Debug çıktılarını ve indirilebilir PDF’i oluşturur.
    """
    if 'file' not in request.files:
        logger.warning("İstek içinde PDF dosyası bulunamadı.")
        return jsonify({"status": "error", "message": "PDF dosyası eksik."}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '' or not uploaded_file.filename.lower().endswith('.pdf'):
        return jsonify({"status": "error", "message": "Geçerli bir PDF dosyası yüklenmedi."}), 400

    # Unique isim
    unique_id = os.urandom(8).hex()
    filename_base = f"processed_document_{unique_id}"
    temp_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename_base}.pdf")
    uploaded_file.save(temp_pdf_path)
    logger.info(f"PDF dosyası kaydedildi: {temp_pdf_path}")

    # --- Reference No çıkarımı (tüm PDF üzerinden) ---
    sender_ref = find_reference_num(temp_pdf_path, use_ocr_fallback=True)

    # Son sayfayı görüntüye çevirme (YOLO için)
    try:
        pages = convert_from_path(temp_pdf_path, dpi=400, poppler_path=POPPLER_PATH)
        if not pages:
            raise ValueError("PDF sayfaları alınamadı.")
        image_np = np.array(pages[-1])  # son sayfa
    except Exception as e:
        logger.error(f"PDF görüntüye çevrilemedi: {e}")
        return jsonify({"status": "error", "message": "PDF işlenemedi."}), 500
    
        # >>>>> PATCH: Açı tespiti ve görüntüyü döndür <<<<<
    try:
        bgr_for_angle = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # son sayfa ile aynı
        orientation_delta = detect_correction_angle(img_bgr=bgr_for_angle)  # 0/90/180/270
        logger.info(f"Detected correction angle: {orientation_delta}°")
    except Exception as e:
        logger.warning(f"Orientation detection failed: {e}")
        orientation_delta = 0

    # YOLO'ya girmeden önce görseli düzelt
    image_np = rotate_rgb_90s(image_np, orientation_delta)

    # Çıktı yolları
    output_pdf = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename_base}.pdf")
    yolo_debug = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename_base}_yolo_debug.jpg")

    # YOLO imza kontrolü, dekont sayısı tespiti
    receipt_count = detect_receipt_count(temp_pdf_path)
    logger.info(f"Tespit edilen dekont sayısı: {receipt_count}")
    is_signed, message, yolo_debug_img = check_signature_logic_yolo(image_np, debug_image_path=yolo_debug, receipt_count=receipt_count)

    ok = process_and_correct_pdf(temp_pdf_path, output_pdf)

    if ok:
        # iş başarıyla bitti; temp'i silebilirsin
        try:
            os.remove(temp_pdf_path)
        except Exception:
            pass
    else:
        logger.warning("process_and_correct_pdf başarısız. Orijinali kopyalıyorum.")
        try:
            shutil.copy2(temp_pdf_path, output_pdf)
        except Exception as e:
            logger.warning(f"PDF çıktı klasörüne kopyalanamadı: {e}")
            output_pdf = None
        finally:
            try:
                os.remove(temp_pdf_path)
            except Exception:
                pass


    return jsonify({
        "status": "success",
        "signed": is_signed,
        "signature_type": "customer" if is_signed else "none",
        "detection_method": "YOLOv8" if (is_signed and yolo_model is not None) else "None",
        "receipt_count": int(receipt_count),
        "message": message,
        "sender_ref": sender_ref,
        "orientation_angle": int(orientation_delta),   # <<--- EKLENDİ
        "download_pdf": os.path.basename(output_pdf) if output_pdf else None,
        "debug_yolo_image": os.path.basename(yolo_debug),
        "download_links": {
            "pdf": f"/download/{os.path.basename(output_pdf)}" if output_pdf else None,
            "yolo_debug_jpg": f"/download/{os.path.basename(yolo_debug)}"
        }
    })

@app.route('/extract-reference', methods=['POST'])
def extract_reference():
    """
    Sadece Reference No (11 hane) çıkarmak için basit test endpoint'i.
    """
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "PDF dosyası eksik."}), 400

    f = request.files['file']
    unique_id = os.urandom(8).hex()
    tmp_pdf = os.path.join(app.config['UPLOAD_FOLDER'], f"ref_{unique_id}.pdf")
    f.save(tmp_pdf)
    try:
        ref = find_reference_num(tmp_pdf, use_ocr_fallback=True)
        return jsonify({"status": "success", "sender_ref": ref})
    finally:
        try:
            os.remove(tmp_pdf)
        except Exception:
            pass

@app.route('/download/<filename>')
def download_file(filename):

    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        logger.error(f"İstenen dosya bulunamadı: {filename}")
        return jsonify({"status": "error", "message": "File not found"}), 404

if __name__ == '__main__':
    # Örn: python app.py
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)  # debug=True sadece geliştirme için
