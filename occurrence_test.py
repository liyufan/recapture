import cv2
import numpy as np
import skimage.io
from skimage.feature import greycomatrix, greycoprops
from sklearn.utils import Bunch


# noinspection DuplicatedCode
def solve():
	img = skimage.io.imread("/home/lyf/data/recapture/all/R/9 (7).JPG")  # 在这里读取图片
	if img is None:
		return -1
	cv2.namedWindow('res', cv2.WINDOW_NORMAL)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	fil = np.array(([-0.25, 0.5, -0.25],
	                [0.5, 0, 0.5],
	                [0, 0, 0]), dtype='float32')

	res = cv2.filter2D(img, -1, fil)
	final = img - res
	fi = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
	cv2.imshow('res', fi)

	# noinspection DuplicatedCode
	co_occurrence = greycomatrix(
		fi, distances=[5], angles=[0, np.pi / 2], levels=256, symmetric=True,
		normed=True)
	contrast = greycoprops(co_occurrence, 'contrast')
	dissimilarity = greycoprops(co_occurrence, 'dissimilarity')
	homogeneity = greycoprops(co_occurrence, 'homogeneity')
	energy = greycoprops(co_occurrence, 'energy')
	correlation = greycoprops(co_occurrence, 'correlation')
	asm = greycoprops(co_occurrence, 'ASM')

	return Bunch(
		contrast=contrast,
		dissimilarity=dissimilarity,
		homogeneity=homogeneity,
		energy=energy,
		correlation=correlation,
		asm=asm
	)


if __name__ == '__main__':
	m = solve()
	print(m.contrast[0])
	cv2.waitKey(0)
	# cv2.destroyAllWindows()
