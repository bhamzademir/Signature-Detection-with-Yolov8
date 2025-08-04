from flask import Flask, request, jsonify
import base64
import io # PDF'i bellekten okumak için
# from your_pdf_signature_library import check_pdf_signature # Kendi imza kontrol fonksiyonunuz

app = Flask(__name__)

@app.route('/check_signature', methods=['POST'])
def check_signature():
    # 1. Content-Type kontrolü
    if request.headers.get('Content-Type') != 'application/json':
        return jsonify({"message": "Unsupported Content-Type. Please send as application/json.", "status": "error"}), 415

    data = request.json # Gelen JSON verisini al

    # 2. Base64 kodlu PDF içeriğini kontrol et
    if not data or 'pdf_content' not in data:
        return jsonify({"message": "No 'pdf_content' found in JSON request.", "status": "error"}), 400

    base64_pdf_string = data['pdf_content']
    file_name = data.get('file_name', 'unknown.pdf') # Dosya adını da alabilirsiniz

    try:
        # 3. Base64 string'i binary PDF verisine dönüştür
        pdf_binary_data = base64.b64decode(base64_pdf_string)

        # 4. Binary veriyi kullanarak imza kontrolü yap
        # Bu kısımda kendi PDF imza doğrulama kütüphanenizi kullanacaksınız.
        # Örneğin: is_signed = check_pdf_signature(pdf_binary_data)
        # Eğer kütüphane dosya yolu bekliyorsa, geçici bir dosyaya yazıp okuyabilirsiniz.
        # Ancak io.BytesIO kullanarak bellekte işlem yapmak genellikle daha iyidir.

        # Örnek olarak basit bir kontrol yapalım:
        # Gerçek uygulamada buraya PDF imza doğrulama mantığınız gelecek
        is_signed = False # Varsayılan olarak imzalı değil
        if b"%PDF" in pdf_binary_data[:100]: # Çok basit bir PDF format kontrolü
            if b"/Sig" in pdf_binary_data: # İmza objesi arama (çok yüzeysel)
                 is_signed = True # Geçici olarak doğru varsayalım

        # 5. Sonucu döndür
        if is_signed:
            return jsonify({"message": f"PDF '{file_name}' is signed.", "status": "success"}), 200
        else:
            return jsonify({"message": f"PDF '{file_name}' is not signed.", "status": "info"}), 200

    except base64.binascii.Error:
        return jsonify({"message": "Invalid Base64 string for pdf_content.", "status": "error"}), 400
    except Exception as e:
        # Diğer olası hataları yakalayın (örn. PDF işleme hataları)
        return jsonify({"message": f"An error occurred: {str(e)}", "status": "error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)