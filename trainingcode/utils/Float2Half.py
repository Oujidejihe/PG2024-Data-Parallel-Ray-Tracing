import numpy as np
import torch

from module import *
from datasets import *
from construct_input import getTestData

mode_set = {
    'test_mode': False,
    'save_mode': True,
    'view_mode': False,
}

# element = "instance_repository"
# obj_name = "CHEVAL_MARLY"
# origin_path = f"../test_data/{element}/originTestData{obj_name}.exr"
# direction_path = f"../test_data/{element}/directionTestData{obj_name}.exr"
#
# width = 960
# height = 540
# data, label = loadNormalizedDatasetsForVIS(origin_path, direction_path)

# width = 100
# height = 100
# data = getTestData(width, height)
# print(data)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

element = "instance_repository"
obj_name = "CHEVAL_MARLY"

origin_vis = f"4Res128{obj_name}SingleOutput_vis"
origin_depth = f"4Res128{obj_name}SingleOutput_depth"

vis_model = torch.jit.load(f"../torchScripts/vis/{origin_vis}.pt")
model = torch.jit.load(f"../torchScripts/depth/{origin_depth}.pt")

print(type(vis_model))
exit(1)
# VIS PART=======
vis_model.eval()
# vis_model.to(torch.float16)
vis_model.to(device)

# DEPTH PART=======
model.eval()
# model.to(torch.float16)
model.to(device)

# SAVE PART=======
np_input = {0.8032, 0.7197, 1.0, 0.8662, 0.5869}
X = torch.Tensor([0.1958, 0.2271, 1.0, 0.7246, 0.85]).reshape((1,5))
# X = X.to(torch.float16)
X = X.to(device)

half_result = vis_model(X)
print(half_result)

# vis_traced_script_module = torch.jit.trace(vis_model, X)
# vis_traced_script_module.save(f'../halfTorchScripts/vis/half_{origin_vis}.pt')
#
# depth_traced_script_module = torch.jit.trace(model, X)
# depth_traced_script_module.save(f'../halfTorchScripts/depth/half_{origin_depth}.pth')

# torch.save(model.state_dict(), 'quantized_model.pth')
# torch.save(model.state_dict(), f'../halfTorchScripts/depth/half_{origin_depth}.pt')



print(half_result)
exit(1)
