import numpy as np
import torch
def getTestData(width, height):
    input_data = torch.zeros(width*height, 5)
    for i in range(height):
        for j in range(width):
            input_data[(width - 1 - i)*width+j][1] = i/height
            input_data[(width - 1 - i)*width+j][0] = j/width
            input_data[(width - 1 - i)*width+j][2] = 0.0
            input_data[(width - 1 - i)*width+j][3] = 0.0
            input_data[(width - 1 - i)*width+j][4] = 0.0

    return input_data
