import cv2
import numpy as np

img = cv2.imread('/home/lyf/data/recapture/all/R/9 (7).JPG')
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
# cv.resizeWindow('img', 500, 500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
i = cv2.imread('/home/lyf/data/recapture/all/R/9 (7).JPG', cv2.IMREAD_GRAYSCALE)
w, h = gray.shape
f = np.zeros([w, h])
print(f.shape, '\n')
print(w, h)
for i in range(0, w - 1):
	for j in range(1, h - 3):
		f[i][j] = gray[i][j - 1] - 3 * gray[i][j] + 3 * gray[i][j + 1] - gray[i][j + 2]
		print(i, '\t', j)

cv2.imshow('img', f)
cv2.waitKey(0)
