import base64
import io
from PIL import Image
import fitz # PyMuPDF
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import logging
from pdf2image import convert_from_path
from ultralytics import YOLO

# Flask uygulamasının log seviyesini ayarlayalım
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Klasör tanımlamaları
UPLOAD_FOLDER = 'temp_files'
OUTPUT_FOLDER = 'output_files' # İndirilebilir ve debug dosyaları için
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Maksimum içerik uzunluğu (örneğin 50 MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Gerekli klasörleri kontrol et ve yoksa oluştur
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Poppler yolu - Kendi sisteminize göre güncelleyin!
# Windows için örnek: r'C:\Program Files\poppler-24.08.0\Library\bin'
# Linux için genellikle kuruluma bağlıdır, örneğin: /usr/bin
POPPLER_PATH = r'C:\Program Files\poppler-24.08.0\Library\bin' # <-- BURAYI KENDİ SİSTEMİNİZE GÖRE AYARLAYIN!

# YOLOv8 model yolu - Kendi model dosyanızın yoluyla değiştirin!
# Örneğin: 'my_yolo_model.pt' veya '/path/to/your/yolov8s-signature-detector.pt'
MODEL_PATH = 'yolov8s.pt' # <-- BURAYI KENDİ MODEL DOSYANIZIN YOLUYLA DEĞİŞTİRİN!

# YOLOv8 modelini global olarak yükle (uygulama başlangıcında bir kere)
yolo_model = None
try:
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
        logger.info(f"YOLOv8 imza tespit modeli yerel yoldan yüklendi: {MODEL_PATH}")
    else:
        logger.error(f"Belirtilen YOLOv8 model dosyası bulunamadı: {MODEL_PATH}. YOLO tabanlı kontrol devre dışı.")
        yolo_model = None
except Exception as e:
    logger.error(f"YOLOv8 modeli yüklenirken hata oluştu: {e}. YOLO tabanlı kontrol devre dışı.")
    yolo_model = None

def base64_to_pdf_and_image(base64_string, temp_filename="temp_doc"):
    """
    Base64 stringini bir PDF dosyasına kaydeder ve ilk sayfasını bir NumPy görüntüsüne dönüştürür.
    Dönüştürülen PDF'in geçici dosya yolunu da döndürür.
    """
    decoded = None
    try:
        decoded = base64.b64decode(base64_string)
    except Exception as e:
        logger.error(f"Base64 çözme hatası: {e}")
        return None, None, "Base64 çözme hatası."

    # Geçici PDF dosyasının yolu
    pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{temp_filename}.pdf")
    try:
        with open(pdf_filepath, "wb") as f:
            f.write(decoded)
        logger.info(f"Geçici PDF dosyası oluşturuldu: {pdf_filepath}")
    except Exception as e:
        logger.error(f"PDF dosyası kaydedilirken hata: {e}")
        # Eğer kaydetme hatası olursa PDF dosya yolu da None olarak döner
        return None, None, "PDF dosyası kaydetme hatası."

    img_np = None
    try:
        pages = convert_from_path(pdf_filepath, dpi=300, poppler_path=POPPLER_PATH)
        if not pages:
            # Sayfa çıkarılamadıysa PDF dosya yolunu yine de döndürür ki silinebilsin
            return None, pdf_filepath, "PDF'den hiçbir sayfa çıkarılamadı."
        img_np = np.array(pages[-1]) # Son sayfayı alıyoruz
        return img_np, pdf_filepath, "Başarılı"
    except Exception as e:
        logger.error(f"PDF'i görüntüye dönüştürürken hata: {e}")
        return None, pdf_filepath, "PDF'i görüntüye dönüştürme hatası."
    finally:
        # Geçici PDF dosyası check_document_signature içinde silinecek
        # Bu fonksiyonda silmiyoruz, çünkü main fonksiyonda dosyayı output_folder'a taşıyacağız
        pass

def check_signature_logic_roi(image_np, debug_image_path=None):
    """
    Belirli bir ROI'de (Region of Interest) imza kontrol mantığı.
    Debug için ROI kutusunu çizip görüntüyü kaydeder.
    """
    if image_np is None:
        return False, "Görüntü boş.", None

    # Görüntü PIL'den geldiği için RGB'dir, OpenCV için BGR'ye dönüştürelim
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img_copy = img_bgr.copy() # Debug için kopyasını alalım
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    x1, y1 = 1400, 1275 # Müşteri İmzası alanı için başlangıç X, Y koordinatı (yaklaşık)
    x2, y2 = 1900, 1425 # Müşteri İmzası alanı için bitiş X, Y koordinatı (yaklaşık)

    # Koordinatların geçerli olup olmadığını kontrol edin
    if not (0 <= y1 < y2 <= img_gray.shape[0] and 0 <= x1 < x2 <= img_gray.shape[1]):
        logger.error(f"Geçersiz ROI koordinatları: ({x1},{y1}) - ({x2},{y2}). Görüntü boyutu: {img_gray.shape}.")
        return False, "Geçersiz ROI koordinatları veya ROI imza alanı dışında.", img_copy

    roi = img_gray[y1:y2, x1:x2]

    # İmza olup olmadığını belirlemek için eşikleme ve kontur analizi
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    roi_thresh_cleaned = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(roi_thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bu eşikleri PDF kalitenize ve imza büyüklüğüne göre ayarlayın
    min_signature_area_threshold = 1000 # Minimum kontur alanı
    min_black_pixels_threshold = 500 # Minimum siyah piksel sayısı

    total_contour_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20: # Küçük gürültüleri filtrele
            total_contour_area += area

    black_pixels = cv2.countNonZero(roi_thresh_cleaned)

    logger.info(f"ROI tabanlı kontrol: Toplam kontur alanı: {total_contour_area}, Siyah piksel: {black_pixels}")

    # Belirlediğimiz eşiklerin üzerinde değerler varsa imzalı kabul et
    signed = (total_contour_area > min_signature_area_threshold) and \
             (black_pixels > min_black_pixels_threshold)

    # Debug için ROI kutusunu çiz
    # Çizim BGR formatında yapılmalı
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2) # Kırmızı kutu (BGR: Mavi, Yeşil, Kırmızı)

    if debug_image_path:
        cv2.imwrite(debug_image_path, img_copy)
        logger.info(f"ROI debug görüntüsü kaydedildi: {debug_image_path}")

    return signed, "ROI tabanlı kontrol tamamlandı.", img_copy # Debug görüntüsünü de döndürüyoruz

def check_signature_logic_yolo(image_np, debug_image_path=None):
    """
    YOLOv8 modeli kullanarak imza tespiti yapar.
    Birden fazla imza bulursa, en sağdaki imzayı müşteri imzası olarak kabul eder.
    Debug için bulunan imzaları kutucuk içine alıp güven skorunu yazar.
    """
    if yolo_model is None:
        logger.error("YOLOv8 modeli yüklenemedi veya hazır değil. İmza tespiti yapılamıyor.")
        return False, "YOLOv8 modeli hazır değil.", None
    if image_np is None:
        return False, "Görüntü boş.", None

    img_bgr_for_drawing = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)

    results = yolo_model(image_np, verbose=False) # YOLO modeli RGB numpy dizilerini doğrudan işleyebilir.

    detected_signatures = []
    
    # Tüm bulunan imzaları toplar ve debug çizimlerini yapar
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls.item()) # Sınıf kimliği
            label = yolo_model.names.get(class_id, f"Sınıf {class_id}") # Etiket adı, yoksa sınıf kimliği
            confidence = float(box.conf.item()) # Güven skoru
            coords = box.xyxy.int().tolist()[0] # [x1, y1, x2, y2] formatında koordinatlar
            x1, y1, x2, y2 = coords
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            logger.info(f"YOLOv8 tespit edildi: Etiket: {label}, Güven: {confidence:.2f}, Koordinatlar: {coords}")

            if label == 'signature' and confidence > 0.3: # Güven eşiğini ayarlayabilirsiniz
                detected_signatures.append({
                    "x_center": x_center,
                    "y_center": y_center,
                    "confidence": confidence,
                    "coords": coords
                })
                # Tüm bulunan imzaları yeşil kutu ile çiz
                cv2.rectangle(img_bgr_for_drawing, (x1, y1), (x2, y2), (0, 255, 0), 2) # Yeşil kutu
                cv2.putText(img_bgr_for_drawing, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    customer_signature_found = False
    message = "YOLOv8 kontrolü tamamlandı, müşteri imzası bulunamadı."

    if detected_signatures:
        if len(detected_signatures) == 1:
            # Tek imza bulunduysa, bu müşteri imzasıdır
            customer_signature_found = True
            message = f"YOLOv8 ile tek imza tespit edildi (Güven: {detected_signatures[0]['confidence']:.2f}). Müşteri imzası olarak kabul edildi."
            logger.info(message)
        else:
            # Birden çok imza bulunduysa, en sağdaki müşteri imzasıdır
            # x_center'a göre azalan sırada sırala (en büyük x_center en sağda)
            detected_signatures.sort(key=lambda s: s['x_center'], reverse=True)
            rightmost_signature = detected_signatures[0] # En sağdaki imza

            customer_signature_found = True
            message = f"YOLOv8 ile birden çok imza tespit edildi. En sağdaki imza müşteri imzası olarak kabul edildi (Güven: {rightmost_signature['confidence']:.2f})."
            logger.info(message)
            
            # Müşteri imzası olarak kabul edilen imzayı özellikle işaretle (örneğin mor kutu)
            x1, y1, x2, y2 = rightmost_signature['coords']
            cv2.rectangle(img_bgr_for_drawing, (x1, y1), (x2, y2), (255, 0, 255), 3) # Mor kutu (daha kalın)
            cv2.putText(img_bgr_for_drawing, "CUSTOMER", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


    if debug_image_path:
        if yolo_model is not None and img_bgr_for_drawing is not None:
            cv2.imwrite(debug_image_path, img_bgr_for_drawing)
            logger.info(f"YOLO debug görüntüsü kaydedildi: {debug_image_path}")
        else:
            logger.warning("YOLO debug görüntüsü kaydedilemedi: model veya görüntü yok.")

    return customer_signature_found, message, img_bgr_for_drawing

@app.route('/check_document_signature', methods=['POST'])
def check_document_signature():
    """
    Kullanıcıdan base64 kodlu PDF/görüntü alır, imza tespiti yapar
    ve sonucu JSON olarak döndürür. Ayrıca debug çıktılarını kaydeder
    ve indirme bağlantıları sağlar.
    """
    data = request.get_json()
    if not data or 'base64' not in data:
        logger.warning("İstek içinde 'base64' verisi bulunamadı.")
        return jsonify({"status": "error", "message": "Missing 'base64' data in request"}), 400

    base64_string = data['base64'].strip()
    if not base64_string:
        logger.warning("Boş base64 stringi alındı.")
        return jsonify({"status": "error", "message": "Empty base64 string provided"}), 400

    # Base64'ten PDF'i ve ilk sayfasını görüntü olarak al
    # pdf_filepath: base64_to_pdf_and_image tarafından oluşturulan geçici PDF'in yolu
    image_np, pdf_filepath, conversion_message = base64_to_pdf_and_image(base64_string)

    if image_np is None:
        logger.error(f"Base64 dönüşüm hatası: {conversion_message}")
        # Hata durumunda geçici PDF dosyası hala varsa sil
        if pdf_filepath and os.path.exists(pdf_filepath):
            os.remove(pdf_filepath)
        return jsonify({"status": "error", "message": f"Base64 dönüşüm hatası: {conversion_message}"}), 500

    # Benzersiz bir dosya adı oluşturmak için (aynı anda birden çok istek gelirse çakışmayı önler)
    unique_id = os.urandom(8).hex()
    filename_base = f"processed_document_{unique_id}"

    # Çıktı dosyaları için yollar
    pdf_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename_base}.pdf")
    roi_debug_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename_base}_roi_debug.jpg")
    yolo_debug_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename_base}_yolo_debug.jpg")

    # 1. Aşama: ROI tabanlı kontrol (debug görüntüsünü kaydetmek için yolu gönder)
    is_signed_roi, message_roi, roi_debug_image = check_signature_logic_roi(image_np.copy(), roi_debug_path)

    final_signed = False
    detection_method = "None"
    final_message = ""
    signature_type = "none"

    if is_signed_roi:
        # Konum doğruysa ve ROI'den imza bulunduysa, YOLO'ya geçmeye gerek yok.
        final_signed = True
        detection_method = "ROI"
        signature_type = "customer"
        final_message = f"ROI tabanlı: Müşteri imzası tespit edildi. {message_roi}"
        logger.info(final_message)
    else:
        # ROI imza bulamazsa veya emin değilse, YOLOv8'e geç.
        logger.info(f"ROI tabanlı kontrol imza bulamadı veya emin değil: {message_roi}. YOLOv8'e geçiliyor.")
        is_signed_yolo, message_yolo, yolo_debug_image = check_signature_logic_yolo(image_np.copy(), yolo_debug_path)
        
        if is_signed_yolo:
            final_signed = True
            detection_method = "YOLOv8"
            signature_type = "customer" # YOLO'nun en sağdaki veya tek imzası müşteri imzası kabul edildi
            final_message = f"YOLOv8 tabanlı: Müşteri imzası tespit edildi. {message_yolo}"
            logger.info(final_message)
        else:
            final_signed = False
            detection_method = "None"
            signature_type = "none"
            final_message = f"Belgede müşteri imzası tespit edilemedi. (ROI: {message_roi}, YOLOv8: {message_yolo})"
            logger.info(final_message)


    # Orijinal PDF dosyasını geçici klasörden çıktı klasörüne taşı (indirme için)
    # Bu, orijinal PDF'in de indirilebilir olmasını sağlar.
    try:
        if pdf_filepath and os.path.exists(pdf_filepath):
            os.rename(pdf_filepath, pdf_output_path) # Dosyayı taşı
            logger.info(f"Orijinal PDF indirilebilir olarak kaydedildi: {pdf_output_path}")
        else:
            pdf_output_path = None # PDF dosyası yoksa indirme bağlantısını boş bırak
            logger.warning("Orijinal PDF dosyası bulunamadı, indirme bağlantısı olmayacak.")
    except Exception as e:
        logger.error(f"PDF dosyasını taşıma hatası: {e}")
        pdf_output_path = None

    # Yanıt oluştur
    response = {
        "status": "success",
        "signed": final_signed,
        "signature_type": signature_type,
        "detection_method": detection_method,
        "message": final_message,
        "download_pdf": os.path.basename(pdf_output_path) if pdf_output_path else None,
        "debug_roi_image": os.path.basename(roi_debug_path),
        "debug_yolo_image": os.path.basename(yolo_debug_path),
        "download_links": {
            "pdf": f"/download/{os.path.basename(pdf_output_path)}" if pdf_output_path else None,
            "roi_debug_jpg": f"/download/{os.path.basename(roi_debug_path)}",
            "yolo_debug_jpg": f"/download/{os.path.basename(yolo_debug_path)}"
        }
    }

    return jsonify(response)

@app.route('/download/<filename>')
def download_file(filename):
    """
    OUTPUT_FOLDER içindeki dosyaların indirilmesini sağlar.
    """
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        logger.error(f"İstenen dosya bulunamadı: {filename}")
        return jsonify({"status": "error", "message": "File not found"}), 404

if __name__ == '__main__':
    # '0.0.0.0' ile dışarıdan erişime açık hale getir, üretimde dikkatli kullanın
    # debug=True geliştirme için uygundur, üretimde False yapılmalıdır.
    app.run(debug=True, host='0.0.0.0', port=5000)