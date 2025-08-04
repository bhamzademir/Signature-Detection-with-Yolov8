from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

def check_signature(pdf_path):

    pages = convert_from_path(pdf_path, dpi=300, poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')

    results = []

    img = np.array(pages[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to grayscale for processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # Kullanıcıdan alınan koordinatlar: sol üst (680,610), sağ alt (975,720)
    x1, y1 = 1400, 1200
    x2, y2 = 2000, 1450

    # ROI'yi renkli resim üzerinde göster
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('selected_roi.jpg', img_with_rect)

    roi = img_gray[y1:y2, x1:x2]

    # Threshold the ROI
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Beyaz pikselleri değil, siyah pikselleri say
    total_pixels = roi_thresh.shape[0] * roi_thresh.shape[1]
    black_pixels = total_pixels - cv2.countNonZero(roi_thresh)

    # Siyah piksel sayısı belli bir eşikten fazlaysa imza var kabul et
    signed = black_pixels > 150
    results.append((1, signed))  # Assuming single page for simplicity

    return results
if __name__ == "__main__":  
    pdf_path = "Dekont.pdf"
    signature_results = check_signature(pdf_path)
    
    for page_num, signed in signature_results:
        status = "Signed" if signed else "Not Signed"
        print(f"Page {page_num}: {status}")

