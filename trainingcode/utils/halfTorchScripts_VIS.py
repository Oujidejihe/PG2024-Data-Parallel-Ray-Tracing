import numpy as np
import torch

from module import *
from datasets import *
from construct_input import getTestData

mode_set = {
    'test_mode': False,
    'save_mode': True,
    'view_mode': True,
}

model_depth = 4
layer_size = 256
element = "instance_repository"
obj_name = "CHEVAL_MARLY"
origin_path = f"../test_data/{element}/originTestData{obj_name}.exr"
direction_path = f"../test_data/{element}/directionTestData{obj_name}.exr"

width = 960
height = 540
data, label = loadNormalizedDatasetsForVIS(origin_path, direction_path)
# data = data.to(torch.float16)
# label = label.to(torch.float16)

# width = 100
# height = 100
# data = getTestData(width, height)
# data = data.to(torch.float16)

# width = 100
# height = 100
# data = getTestData(width, height)
# print(data)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

origin_vis = f"half_{model_depth}Res{layer_size}{obj_name}SingleOutput_vis.pt"
origin_depth = f"half_4Res{layer_size}{obj_name}SingleOutput_depth.pt"

vis_model = torch.jit.load(f"../halfTorchScripts/vis/{origin_vis}")
depth_model = vis_model
# depth_model = torch.jit.load(f"../halfTorchScripts/depth/{origin_depth}")
# VIS PART=======

vis_model.eval()
vis_model.to(device)

VIS_X = data.to(device)
with torch.no_grad():
    VIS_pred = vis_model(VIS_X)
# VIS_pred = VIS_pred.to(torch.float)

VIS_pred0 = torch.squeeze(VIS_pred[:, 0])

VIS_image0 = VIS_pred0.clone().detach()
VIS_image0 = VIS_image0.reshape((VIS_image0.shape[0], 1))
# DEPTH PART=======

depth_model.eval()
depth_model.to(device)

X = data.to(device)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()

for i in range(3):
    start.record()
    with torch.no_grad():
        pred = depth_model(X)
    end.record()

    torch.cuda.synchronize()
    print(start.elapsed_time(end))

pred0 = torch.squeeze(pred[:, 0])
pred = torch.squeeze(pred[:, -1])

# threshold = 0.5
# pred[pred > threshold] = 1.0
# pred[pred < threshold] = 0.0
# print(pred.sum().item())

image0 = pred0.clone().detach()
image0 = image0.reshape((image0.shape[0], 1))

image0 = VIS_image0
# image0[VIS_image0 < 0.6] = 0.0
new_image0 = torch.cat([image0, image0, image0], dim=1)
new_image0 = new_image0.cpu()
np_image0 = np.array(new_image0)
vis_image = np_image0.reshape((height, width, 3))

image = pred.clone().detach()
image = image.reshape((image.shape[0], 1))

# image[VIS_image0 < 0.5] = 0.0

new_image = torch.cat([image, image, image], dim=1)
new_image = new_image.cpu()
np_image = np.array(new_image)
depth_image = np_image.reshape((height, width, 3))
# print(np_image.shape)

if(mode_set['save_mode']):
    cv2.imwrite(f"../test_result/{element}/{obj_name}-{layer_size}-{model_depth}-vis.exr", vis_image)
#     cv2.imwrite(f"../test_result/instance_airplane/airlinear_depth.exr", depth_image)

if (True):
    cv2.imshow('2022-12-17-13-48-model.exr', vis_image)
    cv2.waitKey(0)
