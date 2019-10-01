import torch.nn as nn
import torch.nn.functional as F

from confidnet.models.model import AbstractModel


class SmallConvNetMNISTOODConfid(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.conv1 = nn.Conv2d(config_args["data"]["input_channels"], 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, config_args["data"]["num_classes"])
        self.uncertainty = nn.Linear(128, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.maxpool(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.25, training=self.training)
        else:
            out = self.dropout1(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.mc_dropout:
            out = F.dropout(out, 0.5, training=self.training)
        else:
            out = self.dropout2(out)

        uncertainty = self.uncertainty(out)
        pred = self.fc2(out)
        return pred, uncertainty
