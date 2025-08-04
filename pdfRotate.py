import ocrmypdf
import os

def process_and_correct_pdf(input_pdf_path, output_pdf_path):
    """
    Belgeyi ocrmypdf kullanarak otomatik olarak döndürür, eğriliği düzeltir ve OCR uygular.
    """
    try:
        ocrmypdf.ocr(
            input_pdf_path,
            output_pdf_path,
            rotate_pages=True,
            force_ocr=True,
            rotate_pages_threshold=5.0, # Daha agresif döndürme için eklenebilir
            output_type='pdf'
        )
        print(f"INFO: ocrmypdf: Belge başarıyla işlendi ve kaydedildi: {output_pdf_path}")
        return True
    except ocrmypdf.exceptions.InputFileError as e:
        print(f"HATA: ocrmypdf: Giriş dosyası hatası - {e}")
        return False
    except ocrmypdf.exceptions.FileExistsError as e:
        print(f"HATA: ocrmypdf: Çıkış dosyası zaten var - {e}")
        return False
    except Exception as e:
        print(f"HATA: ocrmypdf: Belge işlenirken bir hata oluştu: {e}")
        return False

# Örnek kullanım (kendi dosya yollarınıza göre ayarlayın)
if __name__ == "__main__":
    # Flask uygulamanızda geçici olarak oluşturduğunuz PDF yolunu buraya verebilirsiniz.
    # Varsayalım ki temp_doc.pdf, işlenmesi gereken yan dönmüş belgeniz.
    input_pdf = "SCAN1046_rotated.pdf" # Burayı kendi dosya yolunuzla değiştirin
    
    # İşlenmiş PDF'i kaydetmek istediğiniz yer
    # Örneğin, output_files klasörünüze kaydedebilirsiniz.
    output_pdf = "output_files\corrected3.pdf" # Burayı da değiştirin

    # Eğer çıkış dosyası zaten varsa, bir hata vermemesi için silebiliriz (opsiyonel)
    if os.path.exists(output_pdf):
        os.remove(output_pdf)

    if process_and_correct_pdf(input_pdf, output_pdf):
        print("Belge döndürüldü ve eğriliği düzeltildi. Şimdi bu yeni PDF'i kullanarak imza tespiti yapabilirsiniz.")
        # Burada, artık "output_pdf" dosyasını kullanarak imza tespitini yeniden çalıştırabilirsiniz.
        # ROI tabanlı kontrolünüzü bu düzeltilmiş PDF üzerinde yapmalısınız.
    else:
        print("Belge işleme başarısız oldu.")