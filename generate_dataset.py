import os
from os import path as osp

root_path = '/data/recapture/'

val_path = osp.join(root_path, 'val')
train_path = osp.join(root_path, 'train')
single_capture_path = osp.join(train_path, 'SingleCaptureImages')
recaptured_path = osp.join(train_path, 'RecapturedImages')

if not osp.exists(single_capture_path):
    os.makedirs(single_capture_path)
if not osp.exists(recaptured_path):
    os.makedirs(recaptured_path)

ori_single_capture_path = osp.join(root_path, 'SingleCaptureImages')
ori_recaptured_path = osp.join(root_path, 'RecapturedImages')

for folder in os.listdir(ori_single_capture_path):
    src_path = osp.join(ori_single_capture_path, folder)
    os.system(f'cp -v {src_path}/* {single_capture_path}/')

for folder in os.listdir(ori_recaptured_path):
    src_path = osp.join(ori_recaptured_path, folder)
    os.system(f'cp -v {src_path}/* {recaptured_path}/')