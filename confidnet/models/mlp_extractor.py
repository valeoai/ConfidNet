import torch.nn as nn
import torch.nn.functional as F

from confidnet.models.model import AbstractModel


class MLPExtractor(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.dropout = config_args["model"]["is_dropout"]
        self.fc1 = nn.Linear(
            config_args["data"]["input_size"][0] * config_args["data"]["input_size"][1],
            config_args["model"]["hidden_size"],
        )

    def forward(self, x):
        out = x.view(-1, self.fc1.in_features)
        if self.dropout:
            out = F.dropout(out, 0.3, training=self.training)
        out = F.relu(self.fc1(out))
        return out
