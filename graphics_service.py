from typing import Tuple 
from PIL import Image as imageMain
from PIL.Image import Image
import tempfile
import pdf2image
import cv2
import numpy as np
import os

class GraphicsService: 
    
    def openImagePil(self, imagePath: str) -> Image:

        return imageMain.open(imagePath) #Open an image using PIL.
    
    def convertPilImageToCvImage(self, pilImage: Image) -> np.ndarray:
        
        return cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)  #Convert a PIL Image to a CV2 image.
    
    def convertCvImageToPilImage(self, cvImage) -> Image:
        
        return imageMain.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))
    
    def openImageCv(self, imagePath: str):
        
        return self.convertPilImageToCvImage(self.openImagePil(imagePath))  #Open an image using OpenCV.

    def cvToGrayScale(self, cvImage):
        
        return cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    
    def cvApplyGaussianBlur(self, cvImage, size: int):
        
        return cv2.GaussianBlur(cvImage, (size, size), 0)
    
    def cvExtractContours(self, cvImage):
        
        contours, hierarchy = cv2.findContours(cvImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours
    
    def paintOverBorder(self, cvImage, borderX: int, borderY: int, color: Tuple[int, int, int]):
        newImage = cvImage.copy()
        height, width, channels = newImage.shape
        for y in range(0, height):
            for x in range(0, width):
                if (y <= borderY) or (height - borderY <= y):
                    newImage[y, x] = color
                if (x <= borderX) or (width - borderX <= x):
                    newImage[y, x] = color
        return newImage

    def rotateImage(self, cvImage, angle: float):
        newImage = cvImage.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return newImage

    def renderPdfDocumentPageToImageFromPath(self, pdfDocPath: str, pageNumber: int, dpi: int) -> str:
        tempFolder = tempfile.gettempdir()
        pageImagePaths = pdf2image.convert_from_path(pdfDocPath, dpi=dpi, output_folder=tempFolder, fmt='png', paths_only=True, thread_count=1, first_page=pageNumber, last_page=pageNumber)
        return pageImagePaths[0]