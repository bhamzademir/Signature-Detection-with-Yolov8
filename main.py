import pytesseract
import cv2
import numpy as np
import os
import tempfile
from pdf2image import convert_from_path

# Kendi servis dosyalarınızı import edin
from graphics_service import GraphicsService
from deskew_service import DeskewService

# Tesseract'ın kurulu olduğundan ve PATH'de olduğundan emin olun veya yolunu belirtin
# Windows için örnek:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DocumentSignatureChecker:
    def __init__(self):
        self.graphics_service = GraphicsService()
        self.deskew_service = DeskewService()

    def check_signature_in_pdf(self, pdf_path: str, debug: bool = False) -> bool:
        """
        PDF belgesindeki "Müşteri İmzası" alanında bir imza olup olmadığını kontrol eder.
        Belgeyi otomatik olarak hizalar ve döndürür.
        """
        temp_dir = None
        img_pil = None # PIL Image nesnesini burada tanımla
        try:
            # Geçici dizin oluştur
            temp_dir = tempfile.mkdtemp()

            # PDF'yi görüntüye dönüştür (ilk sayfa, yüksek DPI)
            print(f"PDF'yi görüntüye dönüştürüyor: {pdf_path}")
            # pdf2image'in geçici dosyaları kaydettiği klasörü belirtiyoruz.
            pages = convert_from_path(pdf_path, dpi=300, output_folder=temp_dir, fmt='png',
                                      first_page=1, last_page=1, thread_count=1)
            
            if not pages:
                print(f"Hata: {pdf_path} dosyasından sayfa dönüştürülemedi.")
                return False

            # Sadece ilk sayfayı işliyoruz
            page_image_path = pages[0] 
            
            # PIL görüntüsünü aç
            img_pil = self.graphics_service.openImagePil(page_image_path)
            # PIL görüntüsünü OpenCV formatına dönüştür
            img_cv = self.graphics_service.convertPilImageToCvImage(img_pil)
            print("Görüntü OpenCV formatına dönüştürüldü.")

            # Eğrilik giderme (Deskew) işlemi
            print("Eğrilik giderme işlemi uygulanıyor...")
            deskewed_img_cv, angle = self.deskew_service.deskew(img_cv)
            print(f"Eğrilik giderildi. Dönüş açısı: {angle:.2f} derece.")

            if debug:
                cv2.imshow("Original Image", img_cv)
                cv2.imshow("Deskewed Image", deskewed_img_cv)
                cv2.waitKey(0)
                cv2.destroyAllWindows() # Pencereyi kapat

            gray_deskewed_img = self.graphics_service.cvToGrayScale(deskewed_img_cv)

            # OCR ile metin konumlarını al
            print("OCR ile 'Müşteri İmzası' etiketi aranıyor...")
            # Pytesseract'a doğrudan OpenCV görüntüsünü veriyoruz
            data = pytesseract.image_to_data(gray_deskewed_img, output_type=pytesseract.Output.DICT, lang='tur')

            found_signature_label = False
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                # 'Müşteri İmzası' kelimesini veya 'Müşteri' ve 'İmzası' kelimelerini arayabiliriz
                if "Müşteri İmzası" in text or ("Müşteri" in text and "İmzası" in text):
                    found_signature_label = True
                    print(f"'{text}' etiketi bulundu.")

                    # Etiketin koordinatları
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                    # İmza alanı genellikle etiketin altında bulunur.
                    # Bu koordinatlar, belgenizin yapısına göre ayarlanmalıdır.
                    # Önerilen: etiketin altından başlayıp belirli bir yükseklik vermek.
                    signature_region_x1 = max(0, x - int(w * 0.1)) # Sol kenardan biraz dışarı
                    signature_region_y1 = max(0, y + h + 10) # Etiketin altından 10 piksel aşağı
                    signature_region_x2 = min(gray_deskewed_img.shape[1], x + w + int(w * 0.5)) # Sağ kenardan biraz dışarı
                    signature_region_y2 = min(gray_deskewed_img.shape[0], signature_region_y1 + 80) # Tahmini imza yüksekliği

                    # İmza bölgesini kırp
                    signature_area = gray_deskewed_img[signature_region_y1:signature_region_y2, signature_region_x1:signature_region_x2]

                    if signature_area.size == 0:
                        print(f"Uyarı: Tespit edilen imza alanı boş veya geçersiz. Koordinatlar kontrol edilsin.")
                        continue # Diğer etiketlere bakmaya devam et

                    # Debug için imza alanını görselleştir
                    if debug:
                        debug_img = deskewed_img_cv.copy()
                        cv2.rectangle(debug_img, (signature_region_x1, signature_region_y1),
                                      (signature_region_x2, signature_region_y2), (0, 255, 0), 2)
                        cv2.imshow("Signature Area for Analysis", debug_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows() # Pencereyi kapat

                    # İmza varlığını kontrol etme (piksel yoğunluğuna göre)
                    # Çok koyu veya çok açık alanları filtrele (beyaz arka plan, siyah metin varsayarak)
                    # Buradaki 220, arka planın beyaz (255) olduğunu ve mürekkebin daha koyu olduğunu varsayar.
                    # Mürekkebi temsil eden pikseller 0-219 aralığında olmalı.
                    # THRESH_BINARY_INV, 220'den küçük değerleri 255 (yani "mürekkep"), 220'den büyükleri 0 (yani "arka plan") yapar.
                    _, thresholded_area = cv2.threshold(signature_area, 220, 255, cv2.THRESH_BINARY_INV) 

                    # İmza alanındaki mürekkep piksel sayısını hesapla
                    # 255 olan pikselleri (önceki eşiklemeye göre mürekkep) sayıyoruz
                    ink_pixels = np.sum(thresholded_area > 0) # > 0 çünkü THRESH_BINARY_INV ile mürekkep 255 oldu
                    total_pixels = thresholded_area.size

                    if total_pixels == 0:
                        print("Uyarı: İmza alanı toplam piksel sayısı sıfır.")
                        return False

                    ink_percentage = (ink_pixels / total_pixels) * 100
                    print(f"İmza alanındaki mürekkep yüzdesi: {ink_percentage:.2f}%")

                    # Bir eşik belirleyin. Bu değer, belgelerinizdeki imza yoğunluğuna göre ayarlanmalıdır.
                    # Örneğin, %0.5'ten fazla mürekkep varsa imza olarak kabul et.
                    signature_threshold_percentage = 0.5

                    if ink_percentage > signature_threshold_percentage:
                        print(f"Dosya: {pdf_path} - Müşteri İmzası tespit edildi.")
                        return True
                    else:
                        print(f"Dosya: {pdf_path} - Müşteri İmzası tespit edilemedi.")
                        return False

            if not found_signature_label:
                print(f"Dosya: {pdf_path} - 'Müşteri İmzası' etiketi bulunamadı.")
                return False

        except Exception as e:
            print(f"PDF işlenirken hata oluştu: {e}")
            return False
        finally:
            # PIL görüntüsünü kapat (önemli!)
            if img_pil:
                img_pil.close()
                del img_pil # Referansı kaldır

            # Geçici dizini temizle
            if temp_dir and os.path.exists(temp_dir):
                print(f"Geçici dizin temizleniyor: {temp_dir}")
                # os.listdir() kullanarak dizindeki tüm dosyaları tek tek sil
                for f in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, f)
                    try:
                        if os.path.isfile(file_path): # Sadece dosyaları sil, alt dizinleri değil
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Dosya silinirken hata oluştu {file_path}: {e}")
                
                # Dizin boşsa sil
                try:
                    os.rmdir(temp_dir)
                    print(f"Geçici dizin başarıyla silindi: {temp_dir}")
                except OSError as e:
                    print(f"Dizin silinirken hata oluştu (boş olmayabilir): {e}")

# Kullanım örneği:
if __name__ == "__main__":
    checker = DocumentSignatureChecker()
    
    # Deneyebileceğiniz PDF dosyasının yolu
    # Örnek: 'yamuk.pdf' dosyanızı bu betiğin yanına koyun
    pdf_document_path = 'signed.pdf' 

    if os.path.exists(pdf_document_path):
        # debug=True görsel pencereleri açar ve hata ayıklamanıza yardımcı olur
        is_signed = checker.check_signature_in_pdf(pdf_document_path, debug=True) 
        print(f"\nPDF '{pdf_document_path}' için imza kontrol sonucu: {'İmzalı' if is_signed else 'İmzasız'}")
    else:
        print(f"Hata: '{pdf_document_path}' dosyası bulunamadı. Lütfen doğru yolu belirtin.")