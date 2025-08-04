import pytesseract
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_path

# Tesseract'ın kurulu olduğundan ve PATH'de olduğundan emin olun veya yolunu belirtin
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Windows için farklı olabilir

def check_signature_in_pdf(pdf_path):
    try:
        # PDF'yi görüntülere dönüştür
        # Yüksek çözünürlük daha iyi OCR sağlar
        pages = convert_from_path(pdf_path, 300)

        for i, page_image in enumerate(pages):
            # PIL görüntüsünü OpenCV formatına dönüştür
            img_cv = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
            gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # OCR ile metin konumlarını al
            data = pytesseract.image_to_data(gray_img, output_type=pytesseract.Output.DICT, lang='tur')

            found_signature_label = False
            for j in range(len(data['text'])):
                text = data['text'][j].strip()
                print(f"Sayfa {i+1}, Metin: '{text}'")
                if "Dalga" in text:
                    found_signature_label = True
                    # Müşteri İmzası başlığının altındaki alanı belirleme
                    # Bu koordinatlar deneme yanılma ile ayarlanabilir
                    x, y, w, h = data['left'][j], data['top'][j], data['width'][j], data['height'][j]

                    # İmzaya ayrılmış bölgeyi tahmin edin (örneğin, başlığın biraz altı ve sağı)
                    # Bu değerler PDF'nizin yapısına göre ayarlanmalıdır
                    signature_region_x = x
                    signature_region_y = y + h + 10 # Başlığın 10 piksel altına
                    signature_region_width = w + 150 # Başlığın genişliğinden biraz daha fazla
                    signature_region_height = 50 # İmza için tahmin edilen yükseklik

                    # Bölgenin resmin sınırları içinde olduğundan emin olun
                    signature_region_x2 = min(signature_region_x + signature_region_width, gray_img.shape[1])
                    signature_region_y2 = min(signature_region_y + signature_region_height, gray_img.shape[0])

                    signature_area = gray_img[signature_region_y:signature_region_y2, signature_region_x:signature_region_x2]
                    print("Zorlu")
                    if signature_area.size == 0:
                        print(f"Uyarı: {pdf_path} - Sayfa {i+1} için imza alanı boş. Bölge tanımı kontrol edilsin.")
                        return False # Veya başka bir hata durumu

                    # İmza alanındaki beyaz olmayan (mürekkep olabilecek) piksellerin yüzdesini hesapla
                    # Eşik değeri (örn. 200), imzanın ne kadar koyu olduğuna bağlıdır.
                    # Daha düşük değerler daha fazla 'mürekkep' pikseli yakalar.
                    non_white_pixels = np.sum(signature_area < 240) # 240, neredeyse beyazı ifade eder

                    # Alanın toplam piksel sayısını hesapla
                    total_pixels = signature_area.size

                    if total_pixels == 0: # Division by zero prevention
                        print(f"Uyarı: {pdf_path} - Sayfa {i+1} için toplam piksel sıfır.")
                        return False

                    ink_percentage = (non_white_pixels / total_pixels) * 100

                    # Bir eşik belirleyin. Bu değer deneme yanılma ile ayarlanmalıdır.
                    # Eğer mürekkep yüzdesi belirli bir eşiğin üzerindeyse, imza var sayılabilir.
                    threshold_percentage = 0.5 # %0.5'ten fazla mürekkep varsa

                    if ink_percentage > threshold_percentage:
                        print(f"Dosya: {pdf_path}, Sayfa: {i+1} - Müşteri İmzası tespit edildi. Mürekkep yüzdesi: {ink_percentage:.2f}%")
                        return True
                    else:
                        print(f"Dosya: {pdf_path}, Sayfa: {i+1} - Müşteri İmzası tespit edilemedi. Mürekkep yüzdesi: {ink_percentage:.2f}%")
                        return False
            
            # Eğer "Müşteri İmzası" başlığı bulunamazsa, bu sayfada imza kontrolü yapamayız
            if not found_signature_label:
                print(f"Dosya: {pdf_path}, Sayfa: {i+1} - 'Müşteri İmzası' etiketi bulunamadı.")
                # İsterseniz burada False döndürebilir veya diğer sayfalara bakmaya devam edebilirsiniz.
                # Bu örnekte, sadece bir imza alanı beklendiği varsayıldı ve bulunamazsa döngüden çıkılır.
                return False

    except Exception as e:
        print(f"Hata oluştu: {e}")
        return False

# Kullanım örneği:
result = check_signature_in_pdf("test_document.pdf")
print(f"İmza kontrol sonucu: {result}")