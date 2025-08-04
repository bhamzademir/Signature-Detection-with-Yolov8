import base64
import io
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import logging
from pdf2image import convert_from_path
from ultralytics import YOLO
import ocrmypdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_files'
OUTPUT_FOLDER = 'output_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

POPPLER_PATH = POPPLER_PATH = r'C:\Program Files\poppler-24.08.0\Library\bin'   # Linux icin gerek yok
MODEL_PATH = 'yolov8s.pt'

yolo_model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def base64_to_pdf_and_image(base64_string, temp_filename="temp_doc"):
    try:
        decoded = base64.b64decode(base64_string)
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        return None, None, "Base64 error"

    pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{temp_filename}.pdf")
    with open(pdf_filepath, "wb") as f:
        f.write(decoded)

    try:
        pages = convert_from_path(pdf_filepath, dpi=300, poppler_path=POPPLER_PATH)
        if not pages:
            return None, pdf_filepath, "No pages found"
        img_np = np.array(pages[-1])
        return img_np, pdf_filepath, "Success"
    except Exception as e:
        logger.error(f"PDF to image error: {e}")
        return None, pdf_filepath, "PDF to image error"

def get_customer_signature_roi(image_np):
    h, w = image_np.shape[:2]
    roi_top = int(h * 0.88)
    roi_bottom = int(h * 0.97)
    roi_left = int(w * 0.52)
    roi_right = int(w * 0.98)
    roi_image = image_np[roi_top:roi_bottom, roi_left:roi_right]
    return roi_image, (roi_left, roi_top)

def detect_signature(image_np):
    results = yolo_model(image_np, verbose=False)
    signatures = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls.item())
            if yolo_model.names.get(cls) != "signature":
                continue
            conf = float(box.conf.item())
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = box.xyxy.int().tolist()[0]
            signatures.append({"coords": (x1, y1, x2, y2), "confidence": conf})
    return signatures

@app.route('/check_document_signature', methods=['POST'])
def check_document_signature():
    data = request.get_json()
    if not data or 'base64' not in data:
        return jsonify({"status": "error", "message": "Missing base64"}), 400

    base64_string = data['base64'].strip()
    unique_id = os.urandom(8).hex()
    filename_base = f"processed_document_{unique_id}"

    image_np, pdf_path, msg = base64_to_pdf_and_image(base64_string, temp_filename=filename_base)
    if image_np is None:
        return jsonify({"status": "error", "message": msg}), 500

    debug_img = image_np.copy()
    yolo_debug_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename_base}_yolo_debug.jpg")

    roi_image, offset = get_customer_signature_roi(image_np)
    roi_h, roi_w = roi_image.shape[:2]
    roi_rect = (offset[0], offset[1], offset[0] + roi_w, offset[1] + roi_h)

    all_signatures = detect_signature(image_np)
    customer_signatures = []

    for sig in all_signatures:
        x1, y1, x2, y2 = sig["coords"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if roi_rect[0] <= center_x <= roi_rect[2] and roi_rect[1] <= center_y <= roi_rect[3]:
            customer_signatures.append(sig)

    if customer_signatures:
        final_signed = True
        best = max(customer_signatures, key=lambda s: s["confidence"])
        x1, y1, x2, y2 = best["coords"]
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        msg = f"Statik ROI içinde imza bulundu. Güven: {best['confidence']:.2f}"
    elif all_signatures:
        final_signed = True
        best = max(all_signatures, key=lambda s: s["confidence"])
        x1, y1, x2, y2 = best["coords"]
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        msg = f"Statik ROI dışında imza bulundu. Güven: {best['confidence']:.2f}"
    else:
        final_signed = False
        msg = "İmza bulunamadı."

    cv2.imwrite(yolo_debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    return jsonify({
        "status": "success",
        "signed": final_signed,
        "message": msg,
        "debug_yolo_image": os.path.basename(yolo_debug_path),
        "download_links": {
            "yolo_debug_jpg": f"/download/{os.path.basename(yolo_debug_path)}"
        }
    })

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
