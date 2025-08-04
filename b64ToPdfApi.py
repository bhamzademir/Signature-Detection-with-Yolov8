import base64
import io
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

def base64_to_pdf_images(base64_string):
	try:
		decoded = base64.b64decode(base64_string)
		pdf_stream = io.BytesIO(decoded)
		doc = fitz.open(stream=pdf_stream, filetype="pdf")
		images = []
		for page in doc:
			pix = page.get_pixmap(dpi=200) # You can adjust DPI as needed
			img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
			images.append(img)
		return images
	except Exception as e:
		print(f"Error converting base64 to PDF images: {e}")
		return None

@app.route('/convert_pdf', methods=['POST'])
def convert_pdf():
	data = request.get_json()
	if not data or 'base64' not in data:
		return jsonify({'error': 'Missing base64 data in request'}), 400

	base64_string = data['base64'].strip()
	images = base64_to_pdf_images(base64_string)

	if images is None:
		return jsonify({'error': 'Failed to process PDF from base64 string. It might be corrupted or not a valid PDF.'}), 500

	if not images:
		return jsonify({'message': 'No images found in the PDF. It might be an empty or image-less PDF.'}), 200
	
    
	output_file_name = "output.pdf"
	try:
		# Create a list of PIL Image objects from numpy arrays
		pil_images = [Image.fromarray(img) for img in images]

		# Save all pages into a single PDF
		if len(pil_images) > 0:
			pil_images[0].save(
				output_file_name,
				"PDF",
				resolution=100.0,
				save_all=True,
				append_images=pil_images[1:]
			)
			return jsonify({'message': f'PDF successfully converted and saved as {output_file_name}'}), 200
		else:
			return jsonify({'message': 'No images to convert to PDF.'}), 200

	except Exception as e:
		return jsonify({'error': f'Error saving output PDF: {e}'}), 500

if __name__ == '__main__':
	app.run(debug=True) # Set debug=False in production