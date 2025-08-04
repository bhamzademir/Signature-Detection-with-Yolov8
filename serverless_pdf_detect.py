import base64
import json
import os
import io # Bellekten dosya okuma/yazma için

# --- 1. Adım: İstemci Tarafı Simülasyonu (PDF'i Base64'e Kodlayıp JSON Hazırlama) ---

def simulate_client_send(pdf_file_path):
    """
    Belirtilen PDF dosyasını Base64 olarak kodlar ve bir JSON objesi hazırlar.
    """
    if not os.path.exists(pdf_file_path):
        print(f"Hata: '{pdf_file_path}' dosyası bulunamadı. Lütfen bir PDF dosyası oluşturun.")
        return None

    try:
        with open(pdf_file_path, "rb") as pdf_file:
            pdf_binary_data = pdf_file.read()

        # PDF'i Base64'e kodla
        base64_encoded_pdf = base64.b64encode(pdf_binary_data).decode('utf-8')

        # JSON objesini hazırla
        payload = {
            "pdf_content": base64_encoded_pdf,
            "file_name": os.path.basename(pdf_file_path),
            "source": "simulated_client"
        }
        print(f"İstemci: '{pdf_file_path}' dosyasını Base64 olarak kodladı ve JSON yükünü hazırladı.")
        return payload

    except Exception as e:
        print(f"İstemci Hatası: PDF okuma veya Base64 kodlama sırasında hata oluştu: {e}")
        return None

# --- 2. Adım: Sunucu Tarafı Simülasyonu (JSON Alıp PDF'i Çözme ve Kontrol) ---

def simulate_server_receive_and_process(request_data):
    """
    Gelen JSON verisini işler, Base64'ü çözer ve PDF imza kontrolünü simüle eder.
    Bu fonksiyon, gerçek bir HTTP endpoint'inin işini yapar.
    """
    if not request_data:
        return {"message": "Boş istek verisi.", "status": "error"}, 400

    # JSON'dan Base64 kodlu PDF içeriğini al
    base64_pdf_string = request_data.get('pdf_content')
    file_name = request_data.get('file_name', 'bilinmeyen_dosya.pdf')

    if not base64_pdf_string:
        return {"message": "JSON'da 'pdf_content' alanı bulunamadı.", "status": "error"}, 400

    try:
        # Base64 string'i binary PDF verisine dönüştür
        pdf_binary_data = base64.b64decode(base64_pdf_string)
        print(f"Sunucu: '{file_name}' ({len(pdf_binary_data)} bayt) dosyasını Base64'ten başarıyla çözdü.")

        # --- İmza Kontrolü Simülasyonu ---
        # Gerçek bir PDF imza doğrulama kütüphanesi (örn. PyPDF2, pypdf, pikepdf) burada kullanılabilir.
        # Şimdilik çok basit bir kontrol yapıyoruz:
        is_signed = False
        if pdf_binary_data.startswith(b"%PDF"): # PDF başlığı kontrolü
            # Çok basit bir simülasyon: "/Sig" veya "/ADBE.PKCS7.DETACHED" gibi imzayla ilgili anahtarlar arayabiliriz.
            # UYARI: Bu gerçek bir imza doğrulaması değildir, sadece anahtar kelime arar!
            if b"/Sig" in pdf_binary_data or b"/ADBE.PKCS7.DETACHED" in pdf_binary_data:
                is_signed = True
            print(f"Sunucu: PDF içeriği üzerinde imza kontrolü simülasyonu yapıldı.")
        else:
            print("Sunucu: Gelen veri PDF formatında görünmüyor (başlık kontrolü başarısız).")

        # Sonucu JSON olarak hazırla
        if is_signed:
            return {"message": f"PDF '{file_name}' imzalı.", "status": "success"}, 200
        else:
            return {"message": f"PDF '{file_name}' imzasız görünüyor.", "status": "info"}, 200

    except base64.binascii.Error:
        return {"message": "Geçersiz Base64 string'i.", "status": "error"}, 400
    except Exception as e:
        return {"message": f"Sunucu Hatası: İşlem sırasında hata oluştu: {e}", "status": "error"}, 500

# --- Ana Çalışma Alanı ---
if __name__ == "__main__":
    pdf_path = "signed.pdf" # Kontrol etmek istediğiniz PDF dosyasının adı

    print("--- Simülasyon Başladı ---")

    # 1. İstemci tarafı PDF'i hazırlar
    json_payload_from_client = simulate_client_send(pdf_path)

    if json_payload_from_client:
        print("\n--- Sunucu Tarafı İşleme Başladı ---")
        # 2. Sunucu tarafı gelen JSON'u işler
        response_json, status_code = simulate_server_receive_and_process(json_payload_from_client)

        print("\n--- Sonuç ---")
        print(f"Durum Kodu: {status_code}")
        print(f"Yanıt: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
    else:
        print("\nSimülasyon iptal edildi.")

    print("\n--- Simülasyon Bitti ---")