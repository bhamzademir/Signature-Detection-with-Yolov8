from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import logging

# Flask uygulamasının log seviyesini ayarlayalım
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_pdfs' # Gelen PDF'leri geçici olarak kaydedeceğimiz klasör
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # Bu ayar doğru ve yerinde

# Upload klasörünü kontrol et ve yoksa oluştur
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Poppler yolu - Kendi sisteminize göre güncelleyin!
POPPLER_PATH = r'C:\Program Files\poppler-24.08.0\Library\bin' # <-- BURAYI KENDİ SİSTEMİNİZE GÖRE AYARLAYIN

def check_signature_logic(pdf_path):
    """
    İmza kontrol mantığını içeren temel fonksiyon.
    PDF yolunu alır ve imzalı olup olmadığını döndürür.
    """
    logger.info(f"PDF dönüştürülüyor: {pdf_path}")
    try:
        # DPI değerini artırarak daha yüksek çözünürlüklü görüntü elde edebiliriz
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        logger.error(f"PDF'i görüntüye dönüştürürken hata: {e}")
        return False, "PDF dönüştürme hatası."

    if not pages:
        logger.warning("PDF'den hiçbir sayfa çıkarılamadı.")
        return False, "PDF'den sayfa bulunamadı."

    img = np.array(pages[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # *******************************************************************
    # BURAYI KESİNLİKLE KENDİ BELGENİZE GÖRE AYARLAYIN!
    # "Müşteri İmzası" alanının koordinatları
    # Bu değerleri boş ve imzalı belge örneklerinizle titizlikle test edin.
    # Imza alanına yazı veya çizgi girmemeli.
    x1, y1 = 1400, 1275  # Örnek değerler, sizin belgenize göre değiştirin!
    x2, y2 = 1900, 1425  # Örnek değerler, sizin belgenize göre değiştirin!
    # *******************************************************************

    # Koordinatların geçerli olup olmadığını kontrol edin
    if not (0 <= y1 < y2 <= img_gray.shape[0] and 0 <= x1 < x2 <= img_gray.shape[1]):
        logger.error(f"Geçersiz ROI koordinatları: ({x1},{y1}) - ({x2},{y2}). Görüntü boyutu: {img_gray.shape}.")
        return False, "Geçersiz ROI koordinatları."

    roi = img_gray[y1:y2, x1:x2]

    # ROI'yi hata ayıklama için kaydet (API'de bu dosyaları tutmamak daha iyidir, sadece geliştirme için)
    # cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'debug_roi_extracted.jpg'), roi)

    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'debug_roi_thresholded.jpg'), roi_thresh)

    kernel = np.ones((2,2),np.uint8)
    roi_thresh_cleaned = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'debug_roi_cleaned.jpg'), roi_thresh_cleaned)

    contours, _ = cv2.findContours(roi_thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Toplam imza alanı veya piksel sayısı için eşik
    # Bu eşik değerleri imzalı/imzasız örneklerle titizlikle ayarlanmalıdır.
    # İmzasız bir belgede 0'a yakın değerler, imzalıda daha yüksek olmalı.
    min_signature_area_threshold = 1000  # Önceki denemelerden
    min_black_pixels_threshold = 500   # Önceki denemelerden

    total_contour_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20: # Küçük gürültüleri filtrele
            total_contour_area += area

    black_pixels = cv2.countNonZero(roi_thresh_cleaned)

    logger.info(f"ROI'deki toplam kontur alanı: {total_contour_area}")
    logger.info(f"ROI'deki toplam siyah piksel sayısı: {black_pixels}")

    signed = (total_contour_area > min_signature_area_threshold) and \
             (black_pixels > min_black_pixels_threshold)

    return signed, "Başarılı" # Dönüş değerini bir tuple yaptık (bool, message)

# BURAYI 'methods=['POST']' OLARAK DÜZELTİN!
@app.route('/check_signature', methods=['POST'])
def check_signature_api():
    if 'pdf_file' not in request.files:
        logger.warning("İstek içinde 'pdf_file' kısmı bulunamadı.")
        return jsonify({"status": "error", "message": "No 'pdf_file' part in the request"}), 400

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        logger.warning("Dosya seçilmedi.")
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if pdf_file:
        filename = secure_filename(pdf_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            pdf_file.save(filepath)
            logger.info(f"Dosya yüklendi: {filepath}")

            # İmza kontrol fonksiyonunu çağır
            is_signed, message = check_signature_logic(filepath)

            # Geçici dosyayı sil
            os.remove(filepath)
            logger.info(f"Geçici dosya silindi: {filepath}")

            return jsonify({"status": "success", "signed": is_signed, "message": message})
        except Exception as e:
            logger.error(f"Dosya işlenirken hata oluştu: {e}")
            # Hata durumunda bile dosyayı silmeye çalış
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"status": "error", "message": f"File processing error: {str(e)}"}), 500

    logger.error("Beklenmedik bir hata oluştu.")
    return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)