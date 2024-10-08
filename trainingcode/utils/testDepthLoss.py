import torch
import cv2
import os
import math
import datetime
import numpy as np
from torch import nn
from datasets import *
from module import *
from torchvision.io import read_image
from params import elements
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
def test(data, label, model, loss_fn):
    size = len(label)
    batch = 12800
    num_batches = int(size / batch)
    model.eval()
    test_loss, correct, depth_loss = 0, 0, 0
    with torch.no_grad():
        for i in range(0, size, batch):
            X = data[i: i + batch, :]
            y = label[i: i + batch]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # pred = torch.squeeze(pred)
            test_loss += loss_fn(pred, y).item()

            vis = pred[:, 0]
            depth = pred[:, 1]

            vis[vis > 0.5] = 1.0
            vis[vis < 0.5] = 0.0

            result = torch.eq(vis, y[:, 0])
            depth_loss += loss_fn(depth, y[:, 1]).item()

            correct += result.sum().item()

    test_loss /= num_batches
    depth_loss /= num_batches
    correct /= size
    print(f"\nTest Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} , depth loss: {depth_loss:>8f} \n", flush=True)

    return test_loss

if __name__ == '__main__':
    setup_seed(19990201)

    instance_name = elements[0]
    building_ID = ""

    print(f"Testing instance {instance_name}!")

    origin_path = f"./geometry_data/San_Miguel/origin{instance_name}.exr"
    direction_path = f"./geometry_data/San_Miguel/direction{instance_name}.exr"

    data, label = loadNormalizedDatasetsBalance(origin_path, direction_path)
    # train_data, train_label, test_data, test_label = getDatasets(data, label)

    model_name = "NeuralVisNetwork128With2ResAndEncoderDoubleOutput"
    model = NeuralVisNetwork2WithResAndEncoderDoubleOutput()

    model_path = f"../module/San_Miguel/2023-04-02-17-19-loss=0.006099-epochs=400-desk10-model.pth"
    model.load_state_dict(torch.load(model_path))

    model.eval()
    model.to(device)

    # loss_fn = nn.BCELoss()
    # loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()

    test_loss = test(data, label, model, loss_fn)
    print("Done!")

