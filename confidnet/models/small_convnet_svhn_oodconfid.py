import torch.nn as nn
import torch.nn.functional as F

from confidnet.models.model import AbstractModel
from confidnet.models.small_convnet_svhn import Conv2dSame


class SmallConvNetSVHNOODConfid(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.conv1 = Conv2dSame(config_args["data"]["input_channels"], 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = Conv2dSame(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = Conv2dSame(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = Conv2dSame(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = Conv2dSame(64, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = Conv2dSame(128, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, config_args["data"]["num_classes"])

        self.uncertainty1 = nn.Linear(512, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv1_bn(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = self.maxpool1(out)
        out = F.dropout(out, 0.3, training=self.training)

        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = self.maxpool2(out)
        out = F.dropout(out, 0.3, training=self.training)

        out = F.relu(self.conv5(out))
        out = self.conv5_bn(out)
        out = F.relu(self.conv6(out))
        out = self.conv6_bn(out)
        out = self.maxpool3(out)
        out = F.dropout(out, 0.3, training=self.training)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, 0.3, training=self.training)

        uncertainty = self.uncertainty1(out)
        pred = self.fc2(out)
        return pred, uncertainty
