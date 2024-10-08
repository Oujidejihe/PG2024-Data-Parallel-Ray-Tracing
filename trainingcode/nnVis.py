import torch
import cv2
from torch import nn
import numpy as np
from datasets import *
from module import *
from params import elements

mode_set = {
    'test_mode': False,
    'save_mode': False,
    'view_mode': True,
}

element = "instance_repository"
obj_name = "CHEVAL_MARLY"
vis_layer_size = 256
depth_layer_size = 256

if (mode_set['test_mode']):
    exit()
else:
    # suffix = "_original_aabb_geo" + str(insID) + ".exr"
    #
    # origin_path = "./test_data/origin" + suffix
    # direction_path = "./test_data/direction" + suffix

    # origin_path = "./test_data/" + instance_name + "/originTestData" + instance_name + f"{building_ID}.exr"
    # direction_path = "./test_data/" + instance_name + "/directionTestData" + instance_name + f"{building_ID}.exr"

    origin_path = f"./test_data/{element}/originTestData{obj_name}.exr"
    direction_path = f"./test_data/{element}/directionTestData{obj_name}.exr"

    data, label = loadNormalizedDatasetsForVIS(origin_path, direction_path)

    # zeros_pad = torch.zeros(5)
    # data[label < 2.5] = zeros_pad


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# VIS PART=======
vis_model = NeuralVisNetworkWith4Res128SingleOutput()
if (vis_layer_size == 128):
    vis_model = NeuralVisNetworkWith4Res128SingleOutput()
elif (vis_layer_size == 256):
    vis_model = NeuralVisNetworkWith4Res256SingleOutput()
elif (vis_layer_size == 512):
    vis_model = NeuralVisNetworkWith4Res512SingleOutput()

vis_model_name = f"NeuralVisNetworkWith6Res{vis_layer_size}SingleOutput"
vis_model_path = f"./module/{element}/vis/NeuralVisNetworkWith6Res256SingleOutput_vis-2024-04-27-06-44-loss=0.005082-epochs=180-CHEVAL_MARLY-model.pth"
vis_model.load_state_dict(torch.load(vis_model_path))

vis_model.eval()
vis_model.to(device)

VIS_X = data.to(device)

VIS_pred = vis_model(VIS_X)

VIS_pred0 = torch.squeeze(VIS_pred[:, 0])

VIS_image0 = VIS_pred0.clone().detach()
VIS_image0 = VIS_image0.reshape((VIS_image0.shape[0], 1))
# DEPTH PART=======
model = NeuralVisNetworkWith4Res128SingleOutput()
if (depth_layer_size == 128):
    model = NeuralVisNetworkWith4Res128SingleOutput()
elif (depth_layer_size == 256):
    model = NeuralVisNetworkWith4Res256SingleOutput()
elif (depth_layer_size == 512):
    model = NeuralVisNetworkWith4Res512SingleOutput()

model_name = f"NeuralVisNetworkWith4Res{depth_layer_size}SingleOutput"
model_path = f"./module/{element}/depth/NeuralVisNetworkWith4Res256SingleOutput_depth-2024-04-24-22-19-loss=0.017192-epochs=0-CHEVAL_MARLY-model.pth"
model.load_state_dict(torch.load(model_path))

model.eval()
model.to(device)

X, y = data.to(device), label.to(device)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()

for i in range(3):
    start.record()
    with torch.no_grad():
        pred = model(X)
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

# image0[VIS_image0 < 0.5] = 0.0
image0 = VIS_image0
new_image0 = torch.cat([image0, image0, image0], dim=1)
new_image0 = new_image0.cpu()
np_image0 = np.array(new_image0)
np_image0 = np_image0.reshape((540, 960, 3))

image = pred.clone().detach()
image = image.reshape((image.shape[0], 1))
new_image = torch.cat([image, image, image], dim=1)
new_image = new_image.cpu()
np_image = np.array(new_image)
np_image = np_image.reshape((540, 960, 3))
# print(np_image.shape)
if(mode_set['save_mode']):
    cv2.imwrite(f"./test_result/{element}/{model_name}-{obj_name}_full_VIS.exr", np_image0)
    cv2.imwrite(f"./test_result/{element}/{model_name}-{obj_name}_full_DEPTH.exr", np_image)

if (mode_set['view_mode']):
    cv2.imshow('2022-12-17-13-48-model.exr', np_image)
    cv2.waitKey(0)
