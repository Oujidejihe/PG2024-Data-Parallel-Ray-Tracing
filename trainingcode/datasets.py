import torch
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import math
import numpy as np
from torch import nn
from params import AABBs
# from torchvision.io import read_image


def loadDatasets(origin_path, direction_path, i):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    np_origin = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
    np_direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)
    print(AABBs[i])

    flatten = nn.Flatten(0, 1)
    origin = torch.tensor(np_origin)
    origin = flatten(origin)

    direction = torch.tensor(np_direction)
    direction = flatten(direction)
    label = direction[:, 2]
    direction = direction[:, :2]

    data = torch.cat([origin, direction], dim=1)

    data[:, 0] = (data[:, 0] - AABBs[i]["minX"]) / (AABBs[i]["maxX"] - AABBs[i]["minX"])
    data[:, 1] = (data[:, 1] - AABBs[i]["minY"]) / (AABBs[i]["maxY"] - AABBs[i]["minY"])
    data[:, 2] = (data[:, 2] - AABBs[i]["minZ"]) / (AABBs[i]["maxZ"] - AABBs[i]["minZ"])
    data[:, 3] = (data[:, 3]) / (2 * math.pi)
    data[:, 4] = (data[:, 4]) / math.pi
    # print(data[:14, :])
    # print(label[:14])
    return data, label

def loadNormalizedDatasetsForVIS(origin_path, direction_path):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    np_origin = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
    np_direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)
    flatten = nn.Flatten(0, 1)
    origin = torch.tensor(np_origin)
    origin = flatten(origin)

    direction = torch.tensor(np_direction)
    direction = flatten(direction)
    label = direction[:, 2]
    direction = direction[:, :2]

    data = torch.cat([origin, direction], dim=1)

    return data, label

def loadNormalizedDatasets(origin_path, direction_path):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    np_origin = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
    np_direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)

    flatten = nn.Flatten(0, 1)
    origin = torch.tensor(np_origin)
    origin = flatten(origin)

    direction = torch.tensor(np_direction)
    direction = flatten(direction)
    label = direction[:, 2]
    direction = direction[:, :2]

    data = torch.cat([origin, direction], dim=1)
    # print(data[:14, :])
    # print(label[:14])

    label_index = np.where(label != 1.0)
    label_index = np.squeeze(np.array(label_index))

    label_equal_index = np.where(label == 1.0)
    label_equal_index = np.squeeze(np.array(label_equal_index))

    equal_data = data[label_equal_index, :]
    equal_label = label[label_equal_index]

    not_equal_data = data[label_index, :]
    not_equal_label = label[label_index]

    random_idx = np.random.permutation(equal_label.shape[0])
    random_idx = random_idx[: int(not_equal_label.shape[0] / 2)]
    equal_data = equal_data[random_idx, :]
    equal_label = equal_label[random_idx]
    # data = torch.squeeze(data)

    data = torch.cat([equal_data, not_equal_data], dim=0)
    label = torch.cat([equal_label, not_equal_label], dim=0)

    print(data.shape)
    print(label.shape)

    print(data[:4, :])
    print(label[:4])
    return data, label
def loadNormalizedDatasetsBalance(origin_path, direction_path):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    np_origin = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
    np_direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)
    radio = 1.0

    flatten = nn.Flatten(0, 1)
    origin = torch.tensor(np_origin)
    origin = flatten(origin)

    direction = torch.tensor(np_direction)
    direction = flatten(direction)
    label = direction[:, 2]
    direction = direction[:, :2]

    data = torch.cat([origin, direction], dim=1)

    # print(data[:14, :])
    # print(label[:14])

    label_index = np.where(label != 1.0)
    label_index = np.squeeze(np.array(label_index))

    label_equal_index = np.where(label == 1.0)
    label_equal_index = np.squeeze(np.array(label_equal_index))

    equal_data = data[label_equal_index, :]
    equal_label = label[label_equal_index]

    not_equal_data = data[label_index, :]
    not_equal_label = label[label_index]

    random_idx = np.random.permutation(equal_label.shape[0])
    random_idx = random_idx[: int(not_equal_label.shape[0]*radio)]
    equal_data = equal_data[random_idx, :]
    equal_label = equal_label[random_idx]
    # data = torch.squeeze(data)

    data = torch.cat([equal_data, not_equal_data], dim=0)
    label = torch.cat([equal_label, not_equal_label], dim=0)

    print(label.shape)
    vis_label = torch.ones(label.shape)
    vis_label[label == 1.0] = 0.0
    label = torch.cat([vis_label.reshape(label.shape[0], 1), label.reshape(label.shape[0], 1)], dim=1)

    return data, label


def loadNormalizedDatasetsBalanceVIS(origin_path, direction_path):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    np_origin = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
    np_direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)
    radio = 1.5

    flatten = nn.Flatten(0, 1)
    origin = torch.tensor(np_origin)
    origin = flatten(origin)

    direction = torch.tensor(np_direction)
    direction = flatten(direction)
    label = direction[:, 2]
    direction = direction[:, :2]

    data = torch.cat([origin, direction], dim=1)

    # print(data[:14, :])
    # print(label[:14])

    label_index = np.where(label != 1.0)
    label_index = np.squeeze(np.array(label_index))

    label_equal_index = np.where(label == 1.0)
    label_equal_index = np.squeeze(np.array(label_equal_index))

    equal_data = data[label_equal_index, :]
    equal_label = label[label_equal_index]

    not_equal_data = data[label_index, :]
    not_equal_label = label[label_index]

    random_idx = np.random.permutation(equal_label.shape[0])
    random_idx = random_idx[: int(not_equal_label.shape[0]*radio)]
    equal_data = equal_data[random_idx, :]
    equal_label = equal_label[random_idx]

    data = torch.cat([equal_data, not_equal_data], dim=0)
    label = torch.cat([equal_label, not_equal_label], dim=0)

    print(label.shape)
    vis_label = torch.ones(label.shape)
    vis_label[label == 1.0] = 0.0

    return data, vis_label

def loadNormalizedDatasetsDepth(origin_path, direction_path):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    np_origin = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
    np_direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)

    flatten = nn.Flatten(0, 1)
    origin = torch.tensor(np_origin)
    origin = flatten(origin)

    direction = torch.tensor(np_direction)
    direction = flatten(direction)
    label = direction[:, 2]
    direction = direction[:, :2]

    data = torch.cat([origin, direction], dim=1)
    # print(data[:14, :])
    # print(label[:14])

    label_index = np.where(label != 1.0)
    label_index = np.squeeze(np.array(label_index))

    label_equal_index = np.where(label == 1.0)
    label_equal_index = np.squeeze(np.array(label_equal_index))

    equal_data = data[label_equal_index, :]
    equal_label = label[label_equal_index]

    not_equal_data = data[label_index, :]
    not_equal_label = label[label_index]

    print(not_equal_label.shape)

    return not_equal_data, not_equal_label
def loadMultiDatasets(origin_path_prefix, direction_path_prefix, size):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    suffix = '.exr'

    empty_data = torch.zeros((1, 6))
    empty_label = torch.zeros((1, ))

    for i in range(size):
        origin_path = origin_path_prefix + str(i) + suffix
        direction_path = direction_path_prefix + str(i) + suffix
        np_origin = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED)
        np_direction = cv2.imread(direction_path, cv2.IMREAD_UNCHANGED)

        flatten = nn.Flatten(0, 1)
        origin = torch.tensor(np_origin)
        origin = flatten(origin)

        direction = torch.tensor(np_direction)
        direction = flatten(direction)
        label = direction[:, 2]
        direction = direction[:, :2]

        instanceID = torch.zeros((label.shape[0], 1), dtype=torch.float)
        instanceID.fill_(i)

        data = torch.cat([origin, direction, instanceID], dim=1)

        data[:, 0] = (data[:, 0] - AABBs[i]["minX"]) / (AABBs[i]["maxX"] - AABBs[i]["minX"])
        data[:, 1] = (data[:, 1] - AABBs[i]["minY"]) / (AABBs[i]["maxY"] - AABBs[i]["minY"])
        data[:, 2] = (data[:, 2] - AABBs[i]["minZ"]) / (AABBs[i]["maxZ"] - AABBs[i]["minZ"])
        data[:, 3] = (data[:, 3]) / (2 * math.pi)
        data[:, 4] = (data[:, 4]) / math.pi
        data[:, 5] = (data[:, 5]) / 4.0

        empty_data = torch.cat([empty_data, data], dim=0)
        empty_label = torch.cat([empty_label, label], dim=0)
        # print(data[:14, :])
        # print(label[:14])
        print(f'{(label.sum().item() / float(label.shape[0])): > 4f}')

    return empty_data, empty_label

def getDatasets(data, label):
    train_ratio = 0.8
    test_ratio = 1.0 - train_ratio

    random_idx = np.random.permutation(data.shape[0])
    new_data = data[random_idx, :]
    new_label = label[random_idx]

    dim1 = new_data.shape[0]
    train_data = new_data[:int(dim1 * train_ratio), :]
    train_label = new_label[:int(dim1 * train_ratio)]

    test_data = new_data[int(dim1 * train_ratio): -1, :]
    test_label = new_label[int(dim1 * train_ratio): -1]

    return train_data, train_label, test_data, test_label

def shuffleDatasets(data, label):
    random_idx = np.random.permutation(data.shape[0])
    new_data = data[random_idx, :]
    new_label = label[random_idx]

    return new_data, new_label
