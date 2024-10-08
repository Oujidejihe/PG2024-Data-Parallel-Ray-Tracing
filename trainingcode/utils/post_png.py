import torch
import cv2
import os
import math
import numpy as np
from torch import nn

image_path = "C:/Users/75277/Desktop/debug/result/res/exr/re-256spp.exr"
direction_path = "C:/Users/75277/Desktop/debug/result/res/exr/directionTestDataRENOMME.exr"

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
np_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
np_direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)

flatten = nn.Flatten(0, 1)
np_image = np.array(flatten(torch.tensor(np_image)))
np_direction = np.array(flatten(torch.tensor(np_direction)))

direction = np_direction[:, 2]
print(direction.shape)

for i in range(np_direction.shape[0]):
    if(direction[i] == 0.0):
        np_image[i][0] = 1.0
        np_image[i][1] = 1.0
        np_image[i][2] = 1.0


np_image = np_image.reshape((540, 960, 3))

cv2.imwrite(f"C:/Users/75277/Desktop/debug/result/res/exr/RENOMME_post_result.exr", np_image)
