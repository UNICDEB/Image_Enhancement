import cv2
import numpy as np

img = cv2.imread("data/GOPR9002.jpg")

# Convert to LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l = clahe.apply(l)

lab = cv2.merge((l,a,b))
enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Sharpen
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpened = cv2.filter2D(enhanced, -1, kernel)

cv2.imwrite("enhanced.jpg", sharpened)
