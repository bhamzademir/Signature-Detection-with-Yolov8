import os
import uuid
import logging
from flask import Flask, request, jsonify
from paddle_ext import run_full_pipeline

# Loglama ayarı
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Geçici dosyaların kaydedileceği klasör
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ocr/scan", methods=["POST"])
def ocr_scan():
    # 1. Dosya Kontrolü (Node.js 'file' key'i ile gönderiyor)
    if 'file' not in request.files:
        logger.error("Request içinde 'file' bulunamadı")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # 2. Dosyayı Geçici Olarak Kaydet
            # Benzersiz bir isim ver ki çakışma olmasın
            ext = os.path.splitext(file.filename)[1]
            if not ext:
                ext = ".jpg" # Varsayılan uzantı
                
            unique_filename = f"{uuid.uuid4()}{ext}"
            temp_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            file.save(temp_path)
            logger.info(f"Dosya kaydedildi: {temp_path}")

            # 3. Pipeline'ı Çalıştır
            # Node.js'ten gelen diğer parametreleri de alabilirsin (örn: docType)
            doc_type = request.form.get('docType', 'UNKNOWN')
            logger.info(f"İşlem başlıyor. DocType: {doc_type}")

            result = run_full_pipeline(
                input_path=temp_path,
                out_dir="ocr_outputs",  # Çıktı klasörü
                debug=False
            )

            # 4. Temizlik (Opsiyonel: İşlem bitince geçici dosyayı sil)
            # os.remove(temp_path) 

            # 5. Sonucu Döndür
            # Frontend'in beklediği formatta dönmek iyi olur
            response = {
                "success": True,
                "result": {
                    "fields": result.get("parsed", {}).get("fields", {}),
                    "raw_text": " ".join(result.get("ocr_raw", {}).get("texts", [])),
                    "docType": doc_type
                }
            }
            return jsonify(response)

        except Exception as e:
            logger.error(f"OCR Hatası: {str(e)}")
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=False)