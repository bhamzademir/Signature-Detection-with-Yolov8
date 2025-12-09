import os
from shutil import copyfile

def senkronize_adlandir(gorsel_klasoru, etiket_klasoru, gorsel_uzantisi=".jpg", etiket_uzantisi=".txt"):
    """
    İki farklı klasördeki (görsel ve etiket) dosyaları senkronize bir şekilde
    1'den başlayarak sıralı olarak yeniden adlandırır (örn: 1.jpg ve 1.txt).
    
    Args:
        gorsel_klasoru (str): Görsel dosyaların bulunduğu klasör yolu.
        etiket_klasoru (str): Etiket dosyalarının bulunduğu klasör yolu.
        gorsel_uzantisi (str): Görsel dosya uzantısı (örn: .jpg).
        etiket_uzantisi (str): Etiket dosya uzantısı (örn: .txt).
    """
    
    # 1. Klasörlerin varlığını ve geçerliliğini kontrol et
    if not os.path.isdir(gorsel_klasoru) or not os.path.isdir(etiket_klasoru):
        print(f"HATA: Görsel ({gorsel_klasoru}) veya Etiket ({etiket_klasoru}) klasörü bulunamadı.")
        return

    # 2. Görsel dosyalarını al (Temel listemiz)
    tum_gorseller = os.listdir(gorsel_klasoru)
    
    # Sadece belirlenen uzantıya sahip görselleri filtrele
    gorsel_dosyalari = [f for f in tum_gorseller 
                        if os.path.splitext(f)[1].lower() == gorsel_uzantisi.lower()]
    
    # Tutarlı bir sıralama için dosya adlarını alfabetik olarak sırala
    gorsel_dosyalari.sort() 

    sayac = 1
    basarili_islem_sayisi = 0

    print(f"Görsel Klasörü: {gorsel_klasoru} | Etiket Klasörü: {etiket_klasoru}")
    print("Senkronize yeniden adlandırma başlatılıyor...")
    print("-" * 60)
    
    for eski_gorsel_ad in gorsel_dosyalari:
        # Etiket dosyasının adını tahmin et (uzantı değişimi)
        kok_ad = os.path.splitext(eski_gorsel_ad)[0]
        eski_etiket_ad = kok_ad + etiket_uzantisi
        
        eski_gorsel_tam_yol = os.path.join(gorsel_klasoru, eski_gorsel_ad)
        eski_etiket_tam_yol = os.path.join(etiket_klasoru, eski_etiket_ad)

        # 3. Etiket dosyasının varlığını kontrol et (Eşleşme kontrolü)
        if not os.path.exists(eski_etiket_tam_yol):
            print(f"  [UYARI] Eşleşen etiket dosyası bulunamadı: {eski_etiket_ad}. Görsel atlanıyor.")
            continue # Eşleşmeyen görselleri atla

        # 4. Yeni isimleri oluştur
        yeni_gorsel_ad = f"{sayac}{gorsel_uzantisi.lower()}"
        yeni_etiket_ad = f"{sayac}{etiket_uzantisi.lower()}"
        
        yeni_gorsel_tam_yol = os.path.join(gorsel_klasoru, yeni_gorsel_ad)
        yeni_etiket_tam_yol = os.path.join(etiket_klasoru, yeni_etiket_ad)

        try:
            # 5. Dosyaları yeniden adlandır
            os.rename(eski_gorsel_tam_yol, yeni_gorsel_tam_yol)
            os.rename(eski_etiket_tam_yol, yeni_etiket_tam_yol)
            
            # print(f"  {eski_gorsel_ad} -> {yeni_gorsel_ad} ve {eski_etiket_ad} -> {yeni_etiket_ad}")
            basarili_islem_sayisi += 1
            sayac += 1
        except Exception as e:
            print(f"  [HATA] Dosya {eski_gorsel_ad} / {eski_etiket_ad} yeniden adlandırılamadı: {e}")
            
    print("-" * 60)
    print(f"✅ İşlem tamamlandı. Toplam {basarili_islem_sayisi} çift (görsel + etiket) yeniden adlandırıldı.")

# --- KULLANIM ÖRNEĞİ ---

if __name__ == "__main__":
    
    # Lütfen kendi klasör yollarınızı buraya girin!
    
    # ÖRNEK: images/train ve labels/train klasörlerini senkronize etme
    GÖRSEL_KLASORU = "yolo_train/images" 
    ETİKET_KLASORU = "yolo_train/labels"
    
    GÖRSEL_UZANTISI = ".jpg" # Görsellerinizin uzantısı
    ETİKET_UZANTISI = ".txt" # Etiketlerinizin uzantısı (YOLO için .txt)

    senkronize_adlandir(GÖRSEL_KLASORU, ETİKET_KLASORU, GÖRSEL_UZANTISI, ETİKET_UZANTISI)