import torch
import cv2
import os
import math
import numpy as np
from torch import nn
from params import AABBs
from torchvision.io import read_image




os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
direction_path = "/media/xsk/Elements/pycharm/PycharmProjects/vis/test_data/San_Miguel/directionTestDataleaves1.exr"

np_direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)

print(np_direction.shape)

label = np_direction[:, :, 2]
label = label.reshape((540, 960, 1))

data = np.concatenate([label, label, label], axis=2)
print(data.shape)

data = data.reshape((540, 960, 3))

cv2.imwrite("/home/open/copy/result/depth_leaves1.exr", data)

# print(data[:14, :])
# print(label[:14])
