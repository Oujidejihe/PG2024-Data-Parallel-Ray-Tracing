import torch
from torch import nn
import torch.nn.functional as F

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# ResBlock512
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(512, 512),
        )

    def forward(self, x):
        out = self.block(x)
        return F.leaky_relu(x + out)

# ResBlock256
class ResBlock256(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(256, 256),
        )

    def forward(self, x):
        out = self.block(x)
        return F.leaky_relu(x + out)

# ResBlock128
class ResBlock128(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(128, 128),
        )

    def forward(self, x):
        out = self.block(x)
        return F.leaky_relu(x + out)

class NeuralVisNetworkWithRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to256 = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
        )
        self.encoding2to256 = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
        )
        self.pre_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            nn.Linear(512, 512),
        )
        self.post_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    # def forward(self, x):
    #     out1 = self.pre_block(x)
    #     out2 = self.res_block(out1)
    #
    #     return self.post_block(out1 + out2)
    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to256(origin)
        output2 = self.encoding2to256(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(self.pre_block(out1))

        return self.post_block(out1 + out2)

class NeuralVisNetwork512WithResAndEncoderDoubleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to256 = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
        )
        self.encoding2to256 = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
        )
        self.pre_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            nn.Linear(512, 512),
        )
        self.post_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    # def forward(self, x):
    #     out1 = self.pre_block(x)
    #     out2 = self.res_block(out1)
    #
    #     return self.post_block(out1 + out2)
    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to256(origin)
        output2 = self.encoding2to256(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(self.pre_block(out1))

        return self.post_block(out1 + out2)

class NeuralVisNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_block = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            nn.Linear(512, 512),
        )
        self.post_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out1 = self.pre_block(x)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetwork2WithoutRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_block = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out1 = self.pre_block(x)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith3Res256AndEncoderDoubleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith4Res256AndEncoderDoubleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetwork2WithRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_block = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        out1 = self.pre_block(x)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetwork2WithResAndEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetwork2WithResAndEncoderDoubleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetwork3WithResAndEncoderDoubleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)


class NeuralVisNetwork4WithResAndEncoderDoubleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith8Res128AndEncoderDoubleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)


class MultiGeoNeuralVisNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding1to256 = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
        )
        self.encoding5to256 = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
        )
        self.pre_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            nn.Linear(512, 512),
        )
        self.post_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    # def forward(self, x):
    #     out1 = self.pre_block(x)
    #     out2 = self.res_block(out1)
    #
    #     return self.post_block(out1 + out2)
    def forward(self, x):
        feature = x[:, 0:5]
        instanceID = x[:, 5:]

        output1 = self.encoding5to256(feature)
        output2 = self.encoding1to256(instanceID)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(self.pre_block(out1))

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith2Res128AndSingleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith4Res128AndSingleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith4Res128AndDoubleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)
class NeuralVisNetworkWith4Res512SingleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to256 = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
        )
        self.encoding2to256 = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
        )
        self.pre_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            nn.Linear(512, 512),
        )
        self.post_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
        )

    # def forward(self, x):
    #     out1 = self.pre_block(x)
    #     out2 = self.res_block(out1)
    #
    #     return self.post_block(out1 + out2)
    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to256(origin)
        output2 = self.encoding2to256(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(self.pre_block(out1))

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith4Res256SingleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith6Res256SingleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith4Res128SingleOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith4Res128SingleOutputSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
            ResBlock128(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
            # nn.LeakyReLU(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)

class NeuralVisNetworkWith4Res256SingleOutputSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding3to64 = nn.Sequential(
            nn.Linear(3, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.encoding2to64 = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
            nn.LeakyReLU(),
        )
        self.res_block = nn.Sequential(
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
            ResBlock256(),
        )
        self.post_block = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
            # nn.LeakyReLU(),
        )

    def forward(self, x):
        origin = x[:, 0:3]
        direction = x[:, 3:5]

        output1 = self.encoding3to64(origin)
        output2 = self.encoding2to64(direction)

        out1 = torch.cat([output1, output2], dim=1)
        out2 = self.res_block(out1)

        return self.post_block(out1 + out2)