import pickle
from pathlib import Path

f = open('/home/lyf/data/recapture/train/SingleCapturedImage/lbp_feature_80.pkl', 'rb')
g = open('/home/lyf/data/recapture/train/RecapturedImage/lbp_feature_80.pkl', 'rb')
o = open('/home/lyf/dataset/train/RecapturedImage/co_occurrence.pkl', 'rb')
dir = Path('/home/lyf/data/recapture/train')
svm = open('svm_model2.pkl', 'rb')
data = pickle.load(svm)
lbp_dic = pickle.load(o)
print(lbp_dic)

'''
folder = [d for d in dir.iterdir() if d.is_dir()]
cur_dir = folder[0]
lbp = []
for i, j in enumerate(cur_dir.iterdir()):
	if j.suffix not in ('.JPG', '.jpg', '.png', '.PNG'):
		continue
	lbp.append(lbp_dic[j])
'''
