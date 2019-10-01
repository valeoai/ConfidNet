import torch.nn as nn
import torch.nn.functional as F

from confidnet.models.model import AbstractModel


class MLPSelfConfid(AbstractModel):
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

        self.uncertainty1 = nn.Linear(config_args["model"]["hidden_size"], 400)
        self.uncertainty2 = nn.Linear(400, 400)
        self.uncertainty3 = nn.Linear(400, 400)
        self.uncertainty4 = nn.Linear(400, 400)
        self.uncertainty5 = nn.Linear(400, 1)

    def forward(self, x):
        out = x.view(-1, self.fc1.in_features)
        out = F.relu(self.fc1(out))
        if self.dropout:
            if self.mc_dropout:
                out = F.dropout(out, 0.3, training=self.training)
            else:
                out = self.fc_dropout(out)

        uncertainty = F.relu(self.uncertainty1(out))
        uncertainty = F.relu(self.uncertainty2(uncertainty))
        uncertainty = F.relu(self.uncertainty3(uncertainty))
        uncertainty = F.relu(self.uncertainty4(uncertainty))
        uncertainty = self.uncertainty5(uncertainty)

        pred = self.fc2(out)
        return pred, uncertainty
