import torch.nn as nn
import torch.nn.functional as F

from confidnet.models.model import AbstractModel


class MLP(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.dropout = config_args["model"]["is_dropout"]
        self.fc1 = nn.Linear(
            config_args["data"]["input_size"][0] * config_args["data"]["input_size"][1],
            config_args["model"]["hidden_size"],
        )
        self.fc2 = nn.Linear(
            config_args["model"]["hidden_size"], config_args["data"]["num_classes"]
        )
        self.fc_dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = x.view(-1, self.fc1.in_features)
        out = F.relu(self.fc1(out))
        if self.dropout:
            if self.mc_dropout:
                out = F.dropout(out, 0.3, training=self.training)
            else:
                out = self.fc_dropout(out)
        out = self.fc2(out)
        return out
