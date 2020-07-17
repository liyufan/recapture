from test_recap import lbp_describe
import numpy as np
import cv2

i = cv2.imread('/home/lyf/data/recapture/all/R/9 (8).JPG')
gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
lbp_param = [(8, 1), (16, 2), (24, 3), (24, 4)]
lbp_feature = np.concatenate([lbp_describe(gray, p, r) for p, r in lbp_param])
print(f'lbp feature: {len(lbp_feature)}: {lbp_feature}')