import os
import random
import shutil
from pathlib import Path


# noinspection DuplicatedCode
def split(file_dir, test_dir, train_dir, pick_number):
	path_dir = os.listdir(file_dir)  # 取图片的原始路径
	file_number = len(path_dir)
	sample = random.sample(path_dir, pick_number)  # 随机选取pick_number数量的样本图片
	print(sample)
	j = 0
	for name in sample:
		j += 1
		shutil.copy(file_dir + name, test_dir + name)
		print(f'Copying {name} to {test_dir}, {j}/{pick_number}')
	j = 0
	for name in path_dir:
		if name in sample:
			continue
		j += 1
		shutil.copy(file_dir + name, train_dir + name)
		print(f'Copying {name} to {train_dir}, {j}/{file_number - pick_number}')


def clear_folder(path):
	path = Path(path)
	if path.exists():
		os.system(f'rm -r {path}')
	os.makedirs(path)


if __name__ == '__main__':
	work_dir = '/home/lyf/dataset/'  # 源图片文件夹路径
	R = work_dir.__add__('all/RecapturedImages/')
	S = work_dir.__add__('all/SingleCaptureImages/')
	R_train = work_dir.__add__('train/RecapturedImage/')
	R_test = work_dir.__add__('test/RecapturedImage/')
	S_train = work_dir.__add__('train/SingleCapturedImage/')
	S_test = work_dir.__add__('test/SingleCapturedImage/')
	_ = [R_train, R_test, S_train, S_test]
	for i in _:
		clear_folder(i)
	split(R, R_test, R_train, 240)
	split(S, S_test, S_train, 150)
