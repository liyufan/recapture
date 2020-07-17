import cv2
import numpy as np
import pywt


def wavelet(img):
	coefs = pywt.wavedec2(img, wavelet='haar', level=3)
	cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coefs
	feature = []
	for mat in (cH3, cV3, cD3, cH2, cV2, cD2, cH1, cV1, cD1):
		feature.append(mat.mean())
		feature.append(mat.std())

	return np.array(feature)


# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.420.8777&rep=rep1&type=pdf
def color_moment(img):
	# Convert BGR to HSV colorspace
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# Split the channels - h,s,v
	h, s, v = cv2.split(hsv)
	# Initialize the color feature
	moments = []
	# N = h.shape[0] * h.shape[1]
	# The first central moment - average
	h_mean = np.mean(h)  # np.sum(h)/float(N)
	s_mean = np.mean(s)  # np.sum(s)/float(N)
	v_mean = np.mean(v)  # np.sum(v)/float(N)
	moments.extend([h_mean, s_mean, v_mean])
	# The second central moment - standard deviation
	h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
	s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
	v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
	moments.extend([h_std, s_std, v_std])
	# The third central moment - the third root of the skewness
	h_skewness = np.mean(abs(h - h.mean()) ** 3)
	s_skewness = np.mean(abs(s - s.mean()) ** 3)
	v_skewness = np.mean(abs(v - v.mean()) ** 3)
	h_third_moment = h_skewness ** (1. / 3)
	s_third_moment = s_skewness ** (1. / 3)
	v_third_moment = v_skewness ** (1. / 3)
	moments.extend([h_third_moment, s_third_moment, v_third_moment])

	return np.array(moments)


if __name__ == '__main__':
	# 读取灰度图
	image = cv2.imread('/home/lyf/dataset/train/SingleCapturedImage/DS-05-0697-S%RX100.JPG')
	image = cv2.resize(image, (448, 448))
	(B, G, R) = cv2.split(image)
	print(B.shape)
	# mean = B.mean()
	# print(mean)

	# http://sina.sharif.ir/~kharrazi/pubs/icip04_1.pdf
	average = np.average(image, axis=(0, 1))
	# print(average)
	cBG = np.corrcoef(B, G)
	cBR = np.corrcoef(B, R)
	cGR = np.corrcoef(G, R)
	print(cBG)

	x = np.linalg.det(B / 255) ** 2
	y = np.linalg.det(G / 255) ** 2
	z = np.linalg.det(R / 255) ** 2
	E1 = y / x
	E2 = y / z
	E3 = x / z

	# C = np.zeros(B.shape, dtype=np.float32)
	# cv2.normalize(src=B, dst=C, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	'''
	plt.figure('二维小波一级变换')
	for i in [B, G, R]:
		coefs = pywt.wavedec2(i, wavelet='haar', level=3)
	cA, (cH, cV, cD) = coefs

	plt.subplot(221), plt.imshow(cA, 'gray'), plt.title("A")
	plt.subplot(222), plt.imshow(cH, 'gray'), plt.title("H")
	plt.subplot(223), plt.imshow(cV, 'gray'), plt.title("V")
	plt.subplot(224), plt.imshow(cD, 'gray'), plt.title("D")
	plt.show()
	'''
	wavelet_feature = np.concatenate([wavelet(channel) for channel in [B, G, R]])
	moment_feature = color_moment(image)
	print(moment_feature)
	color_features = np.concatenate([average, cBR, cGR, cBG, E1, E2, E3, moment_feature], axis=None)
	# print(wavelet_feature)
