from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import os

def check_signature_improved(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')
    results = []

    img = np.array(pages[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # *******************************************************************
    # BURAYI KESİNLİKLE KENDİ BELGENİZE GÖRE AYARLAYIN!
    # selected_roi_updated.jpg'deki "Müşteri İmzası" alanının koordinatları
    # Örneğin:
    x1, y1 = 1400, 1275  # Bu değerleri manuel olarak belirlemelisiniz!
    x2, y2 = 1900, 1425  # Bu değerleri manuel olarak belirlemelisiniz!
    # *******************************************************************

    # ROI'yi renkli resim üzerinde göster (görsel doğrulama için)
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('step_1_roi_marked.jpg', img_with_rect)

    # ROI'yi kes
    roi = img_gray[y1:y2, x1:x2]
    cv2.imwrite('step_2_extracted_roi.jpg', roi) # Kesilen ROI'yi kaydet

    # ROI üzerinde eşikleme
    # İmza siyah olduğu için THRESH_BINARY_INV kullanıldı
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite('step_3_roi_thresholded.jpg', roi_thresh) # Eşiklenmiş ROI'yi kaydet

    # Morfolojik işlemler: Küçük gürültüleri temizle
    kernel = np.ones((2,2),np.uint8)
    roi_thresh_cleaned = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite('step_4_roi_cleaned.jpg', roi_thresh_cleaned) # Temizlenmiş ROI'yi kaydet

    # Kontur bulma
    contours, _ = cv2.findContours(roi_thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Toplam imza alanı veya piksel sayısı için eşik
    min_signature_area_threshold = 1000  # Bu değeri deneyerek ayarlayın (önceki 500'den daha yüksek)
    min_black_pixels_threshold = 500  # Bu değeri de deneyerek ayarlayın (önceki 200'den daha yüksek)

    total_contour_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20: # Daha küçük gürültüleri filtrele (önceki 10'dan daha yüksek)
            total_contour_area += area

    black_pixels = cv2.countNonZero(roi_thresh_cleaned) # Temizlenmiş eşiklenmiş görüntüdeki siyah pikselleri say

    print(f"ROI'deki toplam kontur alanı: {total_contour_area}")
    print(f"ROI'deki toplam siyah piksel sayısı: {black_pixels}")

    # İmza varlığını belirle
    signed = (total_contour_area > min_signature_area_threshold) and (black_pixels > min_black_pixels_threshold)

    results.append((1, signed))

    return results

if __name__ == "__main__":
    #update the pdf file can be choosen here
    pdf_path = os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("İmza kontrolü başlatılıyor...")
    #pdf_path = "signed2.pdf" # PDF dosyanızın adı
    signature_results = check_signature_improved(pdf_path)

    for page_num, signed in signature_results:
        status = "Signed" if signed else "Not Signed"
        print(f"Page {page_num}: {status}")