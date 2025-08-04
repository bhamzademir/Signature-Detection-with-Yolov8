import cv2
from pdf2image import convert_from_path
import numpy as np

img = np.array(convert_from_path("Dekont.pdf", dpi=300, first_page=1, last_page=1)[0], poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Kutuyu sürükle, ENTER'a bas → (x,y,w,h) döner
bbox = cv2.selectROI("İmza Bölgesini Seç", img_bgr, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = bbox
print("Seçilen ROI:", bbox)



    #x1, y1 = 400, 1300
    #x2, y2 = 450, 1400  clean
