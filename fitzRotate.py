import fitz

import PyPDF2

def detect_real_rotation(page):
    # PyPDF2 rotation flag
    flag = page.get("/Rotate") or 0
    # Eğer flag 0 olduğu halde genişlik>yükseklik ise Landscape'tir
    mediabox = page.mediabox
    w, h = float(mediabox.width), float(mediabox.height)

    if flag == 0 and w > h:
        return 270  # genelde bu PDF’ler 270 ile düzeliyor
    return flag


doc = fitz.open("cropped_rotated.pdf")
page = doc.load_page(0)

print("Rotation:", page.rotation)
reader = PyPDF2.PdfReader("cropped_rotated.pdf")
page2 = reader.pages[0]
real_rotation = detect_real_rotation(page2)
print("Real Rotation:", real_rotation)