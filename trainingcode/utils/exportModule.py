import torch
import torchvision
import cv2
from torch import nn
import numpy as np
from datasets import loadDatasets
from module import *


# insID = 3
# suffix = "_original_aabb_geo" + str(insID) + ".exr"
#
# origin_path = "./data2/origin" + suffix
# direction_path = "./data2/direction" + suffix
# # origin_path = "./data2/origin_multi_shadow_1080p.exr"
# # direction_path = "./data2/direction_multi_shadow_1080p.exr"
#
# data, label = loadDatasets(origin_path, direction_path, insID)

# zeros_pad = torch.zeros(5)
# data[label < 2.5] = zeros_pad

model = NeuralVisNetworkWith4Res256SingleOutput()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# model_name = "NeuralVisNetworkWith4Res128SingleOutput"
# file_name = "instance_airplane"

model_path = f"/media/xsk/Elements/pycharm/PycharmProjects/vis/module/San_Miguel/depth/            NeuralVisNetworkWith4Res256SingleOutput_depth-2023-11-29-21-39-loss=0.001521-epochs=100-umbrella-model.pth"
model.load_state_dict(torch.load(model_path))

model.eval()
model.to(device)

np_input = {0.8032, 0.7197, 1.0, 0.8662, 0.5869}
X = torch.Tensor([0.1958, 0.2271, 1.0, 0.7246, 0.85]).reshape((1,5))
# X, y = data.to(device), label.to(device)
X = X.to(device)
# y = model(X)
# print(y)
# 使用 torch.jit.trace 生成 torch.jit.ScriptModule 来跟踪
traced_script_module = torch.jit.trace(model, X)

traced_script_module.save('../torchScripts/depth/4Res256umbrellaSingleOutput_depth.pt')