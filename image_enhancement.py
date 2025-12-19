# import cv2
# import numpy as np

# img = cv2.imread("data/GOPR9002.jpg")

# # Convert to LAB
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# l, a, b = cv2.split(lab)

# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# l = clahe.apply(l)

# lab = cv2.merge((l,a,b))
# enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# # Sharpen
# kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# sharpened = cv2.filter2D(enhanced, -1, kernel)

# cv2.imwrite("enhanced.jpg", sharpened)

####################

from realesrgan import RealESRGAN
from PIL import Image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=2)
model.load_weights('data/RealESRGAN_x2.pth')

img = Image.open('data/color_2025110315301.jpg')
sr_img = model.predict(img)

sr_img.save('super_res.jpg')
