import cv2
import numpy as np
import matplotlib.pyplot as plt
im = cv2.imread("yamuk.jpg") # read image
imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # turn to gray
imOTSU = cv2.threshold(imGray, 0, 1, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)[1] # get threshold with positive pixels as text
coords = np.column_stack(np.where(imOTSU > 0)) # get coordinates of positive pixels (text)
angle = cv2.minAreaRect(coords)[-1] # get a minAreaRect angle
if angle < -45: # adjust angle
    angle = -(90 + angle)
else:
    angle = -angle
# get width and center for RotationMatrix2D
(h, w) = imGray.shape # get width and height of image
center = (w // 2, h // 2) # get the center of the image
M = cv2.getRotationMatrix2D(center, angle, 1.0) # define the matrix
rotated = cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) # apply it