import os
import random
import shutil
from pathlib import Path


def split(file_dir, test_dir, train_dir):
	path_dir = os.listdir(file_dir)  # 取图片的原始路径
	file_number = len(path_dir)
	rate = 0.3  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
	pick_number = int(file_number * rate)  # 按照rate比例从文件夹中取一定数量图片
	sample = random.sample(path_dir, pick_number)  # 随机选取pick_number数量的样本图片
	print(sample)

	for name in sample:
		shutil.copy(file_dir + name, test_dir + name)
	for name in path_dir:
		if name in sample:
			continue
		shutil.copy(file_dir + name, train_dir + name)


def clear_folder(path):
	path = Path(path)
	if path.exists():
		os.system(f'rm -r {path}')
	os.makedirs(path)


if __name__ == '__main__':
	work_dir = '/home/lyf/data/recapture/'  # 源图片文件夹路径
	R = work_dir.__add__('all/R/')
	S = work_dir.__add__('all/S/')
	R_train = work_dir.__add__('train/RecapturedImage/')
	R_test = work_dir.__add__('test/RecapturedImage/')
	S_train = work_dir.__add__('train/SingleCapturedImage/')
	S_test = work_dir.__add__('test/SingleCapturedImage/')
	_ = [R_train, R_test, S_train, S_test]
	for i in _:
		clear_folder(i)
	split(R, R_test, R_train)
	split(S, S_test, S_train)
