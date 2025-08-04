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
import ocrmypdf # ocrmypdf kütüphanesini import ediyoruz

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
        return None, None, "PDF dosyası kaydetme hatası."

    img_np = None
    try:
        # PDF'i görüntüye dönüştür (orijinal hali)
        pages = convert_from_path(pdf_filepath, dpi=300, poppler_path=POPPLER_PATH)
        if not pages:
            return None, pdf_filepath, "PDF'den hiçbir sayfa çıkarılamadı."
        img_np = np.array(pages[-1]) # Son sayfayı alıyoruz
        return img_np, pdf_filepath, "Başarılı"
    except Exception as e:
        logger.error(f"PDF'i görüntüye dönüştürürken hata: {e}")
        return None, pdf_filepath, "PDF'i görüntüye dönüştürme hatası."

def process_and_correct_pdf(input_pdf_path, output_pdf_path):
    """
    Belgeyi ocrmypdf kullanarak otomatik olarak döndürür, eğriliği düzeltir ve OCR uygular.
    """
    try:
        # Çıkış dosyası zaten varsa, ocrmypdf hata vermemesi için silebiliriz
        if os.path.exists(output_pdf_path):
            os.remove(output_pdf_path)
            logger.info(f"Mevcut çıkış dosyası silindi: {output_pdf_path}")

        ocrmypdf.ocr(
            input_pdf_path,
            output_pdf_path,
            rotate_pages=True,
            force_ocr=True,
            rotate_pages_threshold=5.0, # Daha agresif döndürme için eklenebilir
            output_type='pdf'
        )
        logger.info(f"ocrmypdf: Belge başarıyla işlendi ve kaydedildi: {output_pdf_path}")
        return True
    except ocrmypdf.exceptions.InputFileError as e:
        logger.error(f"ocrmypdf: Giriş dosyası hatası - {e}")
        return False
    except ocrmypdf.exceptions.FileExistsError as e:
        logger.error(f"ocrmypdf: Çıkış dosyası zaten var - {e}")
        # Bu hata ocrmypdf'in mevcut dosyaya yazmasını engelleyebilir, yukarıdaki os.remove ile çözülmeli
        return False
    except Exception as e:
        logger.error(f"ocrmypdf: Belge işlenirken bir hata oluştu: {e}")
        return False
    
def is_within_customer_signature_area(x_center, y_center, image_shape):
    """
    Bir imza koordinatının müşteri imzası kutusuna denk gelip gelmediğini kontrol eder.
    """
    h, w = image_shape[:2]
    roi_top = int(h * 0.88)
    roi_bottom = int(h * 0.97)
    roi_left = int(w * 0.52)
    roi_right = int(w * 0.98)
    
    return roi_left <= x_center <= roi_right and roi_top <= y_center <= roi_bottom


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
    
    results = yolo_model(image_np, verbose=False) 

    detected_signatures = []
    
    # Tüm bulunan imzaları toplar ve debug çizimlerini yapar
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls.item()) 
            label = yolo_model.names.get(class_id, f"Sınıf {class_id}")
            confidence = float(box.conf.item())
            coords = box.xyxy.int().tolist()[0]
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
            sig = detected_signatures[0]
            x_c, y_c = sig['x_center'], sig['y_center']

            if is_within_customer_signature_area(x_c, y_c, image_np.shape):
                customer_signature_found = True
                message = f"Tek imza tespit edildi ve müşteri imza alanında (Güven: {sig['confidence']:.2f})."
                logger.info(message)
                # Mor kutu çiz
                x1, y1, x2, y2 = sig["coords"]
                cv2.rectangle(img_bgr_for_drawing, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img_bgr_for_drawing, "CUSTOMER", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                # ROI kutusunu da çiz
                h, w = image_np.shape[:2]
                roi_top = int(h * 0.88)
                roi_bottom = int(h * 0.97)
                roi_left = int(w * 0.52)
                roi_right = int(w * 0.98)
                cv2.rectangle(img_bgr_for_drawing, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)
            else:
                customer_signature_found = False
                message = f"Tek imza bulundu ama müşteri alanında değil (muhtemelen operatör imzası)."
                logger.info(message)
                # ROI kutusunu yine çiz
                h, w = image_np.shape[:2]
                roi_top = int(h * 0.88)
                roi_bottom = int(h * 0.97)
                roi_left = int(w * 0.52)
                roi_right = int(w * 0.98)
                cv2.rectangle(img_bgr_for_drawing, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)

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
    Kullanıcıdan base64 kodlu PDF/görüntü alır, önce döndürür/düzeltir,
    sonra imza tespiti yapar ve sonucu JSON olarak döndürür.
    Debug çıktılarını kaydeder ve indirme bağlantıları sağlar.
    """
    data = request.get_json()
    if not data or 'base64' not in data:
        logger.warning("İstek içinde 'base64' verisi bulunamadı.")
        return jsonify({"status": "error", "message": "Missing 'base64' data in request"}), 400

    base64_string = data['base64'].strip()
    if not base64_string:
        logger.warning("Boş base64 stringi alındı.")
        return jsonify({"status": "error", "message": "Empty base64 string provided"}), 400

    unique_id = os.urandom(8).hex()
    filename_base = f"processed_document_{unique_id}"

    # Orijinal base64 verisinden PDF oluştur
    # image_np: Orijinal PDF'in son sayfasından alınan görüntü
    # pdf_filepath: Orijinal PDF dosyasının geçici yolu
    original_image_np, pdf_filepath, conversion_message = base64_to_pdf_and_image(base64_string, temp_filename=filename_base)

    if original_image_np is None:
        logger.error(f"Base64 dönüşüm hatası: {conversion_message}")
        if pdf_filepath and os.path.exists(pdf_filepath):
            os.remove(pdf_filepath)
        return jsonify({"status": "error", "message": f"Base64 dönüşüm hatası: {conversion_message}"}), 500

    # Düzeltilmiş PDF için geçici bir dosya yolu oluştur
    corrected_pdf_temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename_base}_corrected.pdf")
    
    # PDF'i döndür ve düzelt
    corrected_successfully = process_and_correct_pdf(pdf_filepath, corrected_pdf_temp_path)

    image_for_detection = original_image_np # Varsayılan olarak orijinal görüntüyü kullan

    if corrected_successfully:
        try:
            # Düzeltilmiş PDF'i görüntüye dönüştür
            pages_corrected = convert_from_path(corrected_pdf_temp_path, dpi=300, poppler_path=POPPLER_PATH)
            if pages_corrected:
                image_for_detection = np.array(pages_corrected[-1]) # Düzeltilmiş PDF'in son sayfasını al
                logger.info("Düzeltilmiş PDF görüntüsü imza tespiti için kullanılacak.")
            else:
                logger.warning("Düzeltilmiş PDF'ten görüntü alınamadı, orijinal PDF görüntüsü kullanılacak.")
        except Exception as e:
            logger.error(f"Düzeltilmiş PDF'i görüntüye dönüştürürken hata: {e}. Orijinal PDF görüntüsü kullanılacak.")
    else:
        logger.warning("PDF düzeltme başarısız oldu, orijinal PDF görüntüsü imza tespiti için kullanılacak.")

    # Çıktı dosyaları için yollar
    pdf_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename_base}.pdf")
    # ROI debug görüntüsü için yol tutulmaya devam edebilir (boş kalacak)
    roi_debug_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename_base}_roi_debug.jpg") 
    yolo_debug_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename_base}_yolo_debug.jpg")

    # Sadece YOLOv8 tabanlı kontrolü çağırıyoruz, düzeltilmiş veya orijinal görüntü üzerinde
    is_signed_yolo, message_yolo, yolo_debug_image = check_signature_logic_yolo(image_for_detection.copy(), yolo_debug_path)
    
    final_signed = is_signed_yolo
    detection_method = "YOLOv8" if is_signed_yolo else "None"
    signature_type = "customer" if is_signed_yolo else "none"
    final_message = f"YOLOv8 tabanlı kontrol: {message_yolo}"
    logger.info(final_message)


    # Orijinal PDF dosyasını geçici klasörden çıktı klasörüne taşı (indirme için)
    try:
        if pdf_filepath and os.path.exists(pdf_filepath):
            os.rename(pdf_filepath, pdf_output_path) 
            logger.info(f"Orijinal PDF indirilebilir olarak kaydedildi: {pdf_output_path}")
        else:
            pdf_output_path = None
            logger.warning("Orijinal PDF dosyası bulunamadı, indirme bağlantısı olmayacak.")
    except Exception as e:
        logger.error(f"PDF dosyasını taşıma hatası: {e}")
        pdf_output_path = None

    # Geçici dosyaları temizle
    if os.path.exists(pdf_filepath): # Orijinal geçici PDF
        try:
            os.remove(pdf_filepath)
            logger.info(f"Geçici orijinal PDF silindi: {pdf_filepath}")
        except Exception as e:
            logger.error(f"Geçici orijinal PDF silinirken hata: {e}")
    
    if os.path.exists(corrected_pdf_temp_path): # Düzeltilmiş geçici PDF
        try:
            os.remove(corrected_pdf_temp_path)
            logger.info(f"Geçici düzeltilmiş PDF silindi: {corrected_pdf_temp_path}")
        except Exception as e:
            logger.error(f"Geçici düzeltilmiş PDF silinirken hata: {e}")


    response = {
        "status": "success",
        "signed": final_signed,
        "signature_type": signature_type,
        "detection_method": detection_method,
        "message": final_message,
        "download_pdf": os.path.basename(pdf_output_path) if pdf_output_path else None,
        "debug_roi_image": os.path.basename(roi_debug_path), # Bu dosya oluşmayacak ancak yanıt içinde referans kalabilir
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
    app.run(debug=True, host='0.0.0.0', port=5000)