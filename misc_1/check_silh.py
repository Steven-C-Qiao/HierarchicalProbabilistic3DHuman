import cv2 
import numpy as np

silh = cv2.imread('pred_silh0.png')

print(type(silh))
print(silh.shape)

print(silh[128, :, 0])
print(silh[128, :, 1])
