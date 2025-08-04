import base64
import io
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
import json

def base64_to_image(base64_string):
    decoded = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    return np.array(image)

def base64_to_pdf_images(base64_string):
    decoded = base64.b64decode(base64_string)
    pdf_stream = io.BytesIO(decoded)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        images.append(img)
    return images

# example usage of the base64_to_pdf_images function
if __name__ == "__main__":
    # read base64 from json 
    base64Input = json.load(open('input.json'))['base64']
    base64_string = base64Input.strip()
    images = base64_to_pdf_images(base64_string)
    for i, img in enumerate(images):
        Image.fromarray(img).save(f"page_{i + 1}.png")
        print(f"Saved page {i + 1} as image.")
        # also save as pdf
        img_pil = Image.fromarray(img)
        img_pil.save(f"page_{i + 1}.pdf", "PDF", resolution=100.0)
        print(f"Saved page {i + 1} as PDF.")




