import cv2
import numpy as np

w, h = 500, 500

f = np.zeros([w, h], dtype=np.uint8)
g = np.ones([w, h], dtype=np.uint8)

x = f - g
cv2.imshow('x', x)

cv2.waitKey(0)