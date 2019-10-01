import torch.nn as nn
import torch.nn.functional as F

from confidnet.models.model import AbstractModel


class SmallConvNetMNISTExtractor(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.conv1 = nn.Conv2d(config_args["data"]["input_channels"], 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.maxpool(out)
        out = F.dropout(out, 0.25, training=self.training)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, 0.5, training=self.training)
        return out
