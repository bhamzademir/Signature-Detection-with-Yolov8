from typing import List, Tuple
import cv2
import numpy as np
from graphics_service import GraphicsService

class DeskewService:

    def getSkewAngle(self, cvImage, debug: bool = False) -> float:
        """
        Görüntünün eğim açısını hesaplar.
        :param cvImage: OpenCV formatında görüntü.
        :param debug: Debug modu, True ise görselleştirme yapar.
        :return: Eğim açısı (derece cinsinden).
        """
        newImage = cvImage.copy()
        gray = GraphicsService().cvToGrayScale(newImage)
        blur = GraphicsService().cvApplyGaussianBlur(gray, 9)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        """if debug:
            cv2.imshow ("Thresholded Image", thresh)
            cv2.imshow("Blurred Image", blur)
            cv2.imshow("Gray Image", gray)
            cv2.waitKey()"""

        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)) #30,5 
        dilated = cv2.dilate(thresh, kernel, iterations=5)
        """if debug:
            cv2.imshow("Dilated Image", dilated)
            cv2.waitKey()"""

        contours = GraphicsService().cvExtractContours(dilated)
        """if debug:
            cv2.drawContours(newImage.copy(), contours, -1, (0, 255, 0), 2)
            cv2.imshow("Contours", newImage)
            cv2.waitKey()"""

        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)

        """if debug:
            minAreaRectContour = np.int0(cv2.boxPoints(minAreaRect))
            temp2 = cv2.drawContours(newImage.copy(), [minAreaRectContour], -1, (255, 0, 0), 2)
            cv2.imshow('Largest Contour', temp2)
            cv2.waitKey()"""

        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
            return -1.0 * angle
        elif angle > 45:
            angle = 90 - angle
            return angle
        return -1.0 * angle
    
    def deskew(self, cvImage) -> Tuple:
        angle = self.getSkewAngle(cvImage)
        return GraphicsService().rotateImage(cvImage, -1.0 * angle), angle