from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def check_signature_improved(pdf_path):
    if not os.path.exists(pdf_path):
        messagebox.showerror("Hata", f"Dosya bulunamadı: {pdf_path}")
        return [(1, False)] # Dosya bulunamazsa imzalanmamış olarak dön

    try:
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')
    except Exception as e:
        messagebox.showerror("Hata", f"PDF'i görüntüye dönüştürürken hata oluştu: {e}\nPoppler yolu doğru ayarlanmış mı?")
        return [(1, False)]

    if not pages:
        messagebox.showinfo("Bilgi", "PDF'den hiçbir sayfa çıkarılamadı.")
        return [(1, False)]

    results = []

    img = np.array(pages[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # *******************************************************************
    # BURAYI KESİNLİKLE KENDİ BELGENİZE GÖRE AYARLAYIN!
    # "Müşteri İmzası" alanının koordinatları
    # Bu değerler önceki denemelerinize göre ayarlanmıştır.
    # Kendi belge örneğinizle test ederek optimize etmeye devam etmelisiniz.
    x1, y1 = 1400, 1275
    x2, y2 = 1900, 1425
    # *******************************************************************

    # Hata ayıklama görselleri için bir klasör oluştur
    debug_folder = "debug_images"
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    # ROI'yi renkli resim üzerinde göster (görsel doğrulama için)
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(debug_folder, 'step_1_roi_marked.jpg'), img_with_rect)

    # ROI'yi kes
    roi = img_gray[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(debug_folder, 'step_2_extracted_roi.jpg'), roi) # Kesilen ROI'yi kaydet

    # ROI üzerinde eşikleme
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(debug_folder, 'step_3_roi_thresholded.jpg'), roi_thresh) # Eşiklenmiş ROI'yi kaydet

    # Morfolojik işlemler: Küçük gürültüleri temizle
    kernel = np.ones((2,2),np.uint8)
    roi_thresh_cleaned = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(os.path.join(debug_folder, 'step_4_roi_cleaned.jpg'), roi_thresh_cleaned) # Temizlenmiş ROI'yi kaydet

    # Kontur bulma
    contours, _ = cv2.findContours(roi_thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Toplam imza alanı veya piksel sayısı için eşik
    # Bu değerleri boş ve imzalı belgelerle test ederek daha da optimize etmelisiniz.
    min_signature_area_threshold = 1000
    min_black_pixels_threshold = 500

    total_contour_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20: # Daha küçük gürültüleri filtrele
            total_contour_area += area

    black_pixels = cv2.countNonZero(roi_thresh_cleaned)

    print(f"ROI'deki toplam kontur alanı: {total_contour_area}")
    print(f"ROI'deki toplam siyah piksel sayısı: {black_pixels}")

    # İmza varlığını belirle
    signed = (total_contour_area > min_signature_area_threshold) and (black_pixels > min_black_pixels_threshold)

    results.append((1, signed))

    return results

def browse_pdf():
    # PDF dosyası seçme iletişim kutusunu aç
    file_path = filedialog.askopenfilename(
        title="PDF Dosyası Seçin",
        filetypes=[("PDF Dosyaları", "*.pdf")]
    )
    if file_path:
        # Seçilen dosyayı etikette göster
        pdf_path_label.config(text=f"Seçilen Dosya: {os.path.basename(file_path)}")
        pdf_path_label.pdf_path = file_path # Dosya yolunu label objesine ekle
        check_button.config(state=tk.NORMAL) # Kontrol butonunu aktif et
    else:
        pdf_path_label.config(text="Lütfen bir PDF dosyası seçin.")
        check_button.config(state=tk.DISABLED) # Kontrol butonunu pasif et

def perform_check():
    pdf_path = getattr(pdf_path_label, 'pdf_path', None)
    if pdf_path:
        print("İmza kontrolü başlatılıyor...")
        signature_results = check_signature_improved(pdf_path)

        for page_num, signed in signature_results:
            status = "İmzalı" if signed else "İmzalanmamış"
            result_label.config(text=f"Sayfa {page_num}: {status}",
                                fg="green" if signed else "red")
            print(f"Sayfa {page_num}: {status}")
    else:
        messagebox.showwarning("Uyarı", "Lütfen önce bir PDF dosyası seçin.")


# Tkinter GUI oluşturma
root = tk.Tk()
root.title("PDF İmza Kontrol Aracı")
root.geometry("400x250") # Pencere boyutunu ayarla

# PDF seçme butonu
browse_button = tk.Button(root, text="PDF Seç", command=browse_pdf)
browse_button.pack(pady=10)

# Seçilen PDF yolunu gösteren etiket
pdf_path_label = tk.Label(root, text="Lütfen bir PDF dosyası seçin.", wraplength=350)
pdf_path_label.pack(pady=5)
pdf_path_label.pdf_path = None # Başlangıçta dosya yolu yok

# İmza kontrol butonu
check_button = tk.Button(root, text="İmzayı Kontrol Et", command=perform_check, state=tk.DISABLED)
check_button.pack(pady=10)

# Sonucu gösteren etiket
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

root.mainloop()