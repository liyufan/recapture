import argparse
import os
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage.io
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import Bunch


# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.420.8777&rep=rep1&type=pdf
# noinspection DuplicatedCode
def color_moment(img):
	# Convert RGB to HSV colorspace
	# 调用此方法时为skimage读入图片
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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


# noinspection DuplicatedCode
def prepare_image_feature(container_path):
	image_dir = Path(container_path)  # /train or /test
	# folders = [directory for directory in image_dir.iterdir() if
	# directory.is_dir()]  # 文件夹列表：[/RecapturedImage, /SingleCapturedImage]
	# TODO: Change name here
	class_array = ['RecapturedImage', 'SingleCapturedImage']
	folders = []
	for cls in class_array:
		p = image_dir.joinpath(cls)
		folders.append(p)
	categories = [fo.name for fo in folders]  # 文件夹名称列表：['Recapture..', 'singleCapture..']
	description = "A image classification dataset"
	color_hist_data = []
	target = []
	file_names = []
	file = Path()
	for label, direc in enumerate(folders):  # label 0，1为文件夹下标，direc为文件夹：/RecapturedImage /Single..
		color_file_path = image_dir.joinpath(direc, 'color.pkl')  # 生成pkl文件
		if color_file_path.exists() and color_file_path.stat().st_size > 0:
			with open(color_file_path, 'rb') as cache_file:
				color_dic = pkl.load(cache_file)
		else:
			color_dic = dict()
		dir_stack = [direc]
		while len(dir_stack):
			cur_dir = dir_stack.pop()  # 当前文件夹
			print(f'iter into {cur_dir.name}')
			for j, dir in enumerate(cur_dir.iterdir()):
				if dir.is_dir():  # /recapture里面还有文件夹
					continue
				elif dir.is_file():  # 检查到文件，判断是否是图片
					file = dir
				if file.suffix not in ('.JPG', '.jpg', '.png', '.PNG'):
					print(f'skip non image file {file}')
					continue
				if file in color_dic:
					color_hist_data.append(color_dic[file])
					target.append(label)
					file_names.append(file)
					continue
				print(f'start reading {j}: {file}')
				image = skimage.io.imread(file)
				image = cv2.resize(image, (448, 448))
				(R, G, B) = (image[:, :, 0], image[:, :, 1], image[:, :, 2])

				# http://sina.sharif.ir/~kharrazi/pubs/icip04_1.pdf
				average = np.average(image, axis=(0, 1))

				cbg = np.corrcoef(B, G).mean() + 1e-7
				cbr = np.corrcoef(B, R).mean() + 1e-7
				cgr = np.corrcoef(G, R).mean() + 1e-7

				x = np.linalg.det(B / 255) ** 2 + 1e-7
				y = np.linalg.det(G / 255) ** 2 + 1e-7
				z = np.linalg.det(R / 255) ** 2 + 1e-7
				e1 = y / x
				e2 = y / z
				e3 = x / z

				moment_feature = color_moment(image)

				# TODO: add feature here
				# color_feature = np.concatenate([average, [cbg, cbr, cgr, e1, e2, e3], moment_feature])
				color_feature = np.concatenate([average, moment_feature])
				color_dic[file] = color_feature

				print(f'dumping...')
				with open(color_file_path, 'wb') as cache_file:
					pkl.dump(color_dic, cache_file)

				print(f'color feature: {len(color_feature)}: {color_feature}')
				color_hist_data.append(color_feature)
				target.append(label)
				file_names.append(file)

	color_hist_data = np.array(color_hist_data)
	target = np.array(target)

	return Bunch(
		color_hist_data=color_hist_data,
		target=target,
		target_names=categories,
		file_list=file_names,
		DESCR=description)


def clear_folder(path):
	if path.exists():
		os.system(f'rm -r {path}')
	os.makedirs(path)


def test_all(test_path, model_path):
	test_path = Path(test_path)
	res_path = test_path.parent.joinpath('res')  # res文件夹为SVM按预测类别整理的
	# TODO: Change name here
	class_array = ['RecapturedImage', 'SingleCapturedImage']
	dst_paths = []
	for cls in class_array:  # 此循环只用于清空上一次预测的目录
		path = res_path.joinpath(cls)
		clear_folder(path)
		dst_paths.append(path)
	with open(model_path, 'rb') as f:
		clf = pkl.load(f)  # clf为SVM参数
	test_dataset = prepare_image_feature(test_path)
	preds = clf.predict(test_dataset.color_hist_data)  # 为test文件夹所有图片的预测类别
	print(preds)
	print(test_dataset.target)  # 为test文件夹所有图片的真实类别---[0 0 0 0 0 ... 1 1 1 1 1 ...]
	print(test_dataset.target_names)  # ['RecapturedImage', 'SingleCapturedImage']
	print(test_dataset.file_list)
	print(f'Pred: GroundTruth: filepath')
	for pred_label, name, gnd in zip(preds, test_dataset.file_list, test_dataset.target):
		print(f'{pred_label}:\t\t{gnd}:\t\t{name}')  # 预测类别 实际类别 文件名
	# os.system(f'cp "{name}" "{dst_paths[pred_label]}/"')
	print(
		f"Classification report - \n{metrics.classification_report(test_dataset.target, preds, target_names=class_array)}\n")
	print("Confusion matrix -\n")
	print(pd.crosstab(pd.Series(test_dataset.target, name='Actual'), pd.Series(preds, name='Predicted')))
	return preds


def parse_args():
	parser = argparse.ArgumentParser(
		description='Train a network with Detectron'
	)
	parser.add_argument(
		'--testall',
		dest='test_all',
		help='test all images',
		action='store_true'
	)
	parser.add_argument(
		'--retrain',
		dest='retrain',
		help='retrain the model',
		action='store_true'
	)
	return parser.parse_args()


if __name__ == '__main__':
	data_path = '/home/lyf/dataset/train'
	test_data_path = '/home/lyf/dataset/test'
	# TODO: Change svm model name
	model_path = './svm_color.pkl'
	# TODO: Change name here
	class_array = ['RecapturedImage', 'SingleCapturedImage']
	retrain = True
	args = parse_args()

	if args.retrain:
		image_dataset = prepare_image_feature(data_path)
		X_train, X_val, y_train, y_val = train_test_split(image_dataset.color_hist_data, image_dataset.target,
		                                                  test_size=0.3)

		c_range = np.logspace(-5, 15, 11, base=2)
		gamma_range = np.logspace(-9, 3, 13, base=2)
		param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
		svc = svm.SVC(kernel='rbf', class_weight='balanced')
		grid = GridSearchCV(svc, param_grid, n_jobs=-1, cv=3)
		clf = grid.fit(X_train, y_train)
		score = grid.score(X_val, y_val)
		print('the score is %s' % score)
		with open(model_path, 'wb') as f:
			pkl.dump(clf, f)
		y_pred = clf.predict(X_val)
		print(
			f"Classification report - \n{clf}:\n{metrics.classification_report(y_val, y_pred, target_names=class_array)}\n")
		print("Confusion matrix -\n")
		print(pd.crosstab(pd.Series(y_val, name='Actual'), pd.Series(y_pred, name='Predicted')))
	if os.path.exists(model_path):
		if args.test_all:
			preds = test_all(test_data_path, model_path)
