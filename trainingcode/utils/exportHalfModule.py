import torch
import torchvision
import cv2
from torch import nn
import numpy as np
from datasets import loadDatasets
from module import *

NN_type = "vis"
element = "instance_repository"
origin_pth_name = "NeuralVisNetworkWith4Res256SingleOutput_vis-2024-04-27-19-04-loss=0.004437-epochs=120-CHEVAL_MARLY-model.pth"
obj_name = "CHEVAL_MARLY"
model_depth = 4
layer_size = 256

model = NeuralVisNetworkWith4Res128SingleOutput()
if (model_depth == 4):
    if (layer_size == 128):
        model = NeuralVisNetworkWith4Res128SingleOutput()
    elif (layer_size == 256):
        model = NeuralVisNetworkWith4Res256SingleOutput()
    elif (layer_size == 512):
        model = NeuralVisNetworkWith4Res512SingleOutput()
elif (model_depth == 6):
    if (layer_size == 128):
        # model = NeuralVisNetworkWith6Res128SingleOutput()
        exit()
    elif (layer_size == 256):
        model = NeuralVisNetworkWith6Res256SingleOutput()
    elif (layer_size == 512):
        # model = NeuralVisNetworkWith6Res512SingleOutput()
        exit()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

np_input = torch.tensor([0.5, 0.5, 0.8, 1.0, 1.0]).resize_(1, 5)
X = np_input
X = X.to(device)

model_path = f"../module/{element}/{NN_type}/{origin_pth_name}"
model.load_state_dict(torch.load(model_path))

model.eval()
model.to(device)
#
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
#
# torch.cuda.synchronize()
#
# for i in range(3):
#     start.record()
#     with torch.no_grad():
#         pred = model(X)
#     end.record()
#
#     torch.cuda.synchronize()
#     print(start.elapsed_time(end))
#
#
# full_result = model(X)
# print(y)
# X, y = data.to(device), label.to(device)

# model = model.to(torch.float16)
# X = X.to(torch.float16)

# torch.cuda.synchronize()
# for i in range(3):
#     start.record()
#     with torch.no_grad():
#         pred = model(X)
#     end.record()
#
#     torch.cuda.synchronize()
#     print(start.elapsed_time(end))

# half_result = model(X)
#
# half_result = torch.squeeze(half_result[:, 0])
# full_result = torch.squeeze(full_result[:, 0])
#
# MSE = torch.mean(full_result - half_result)
# print(MSE)
#
# exit(1)

# 使用 torch.jit.trace 生成 torch.jit.ScriptModule 来跟踪
traced_script_module = torch.jit.trace(model, X)
traced_script_module.save(f'../halfTorchScripts/{NN_type}/half_{model_depth}Res{layer_size}{obj_name}SingleOutput_{NN_type}.pt')