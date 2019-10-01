from confidnet.models.mlp import MLP
from confidnet.models.mlp_selfconfid import MLPSelfConfid
from confidnet.models.model import AbstractModel


class MLPSelfConfidCloning(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.pred_network = MLP(config_args, device)

        # Small trick to set num classes to 1
        temp = config_args["data"]["num_classes"]
        config_args["data"]["num_classes"] = 1
        self.uncertainty_network = MLPSelfConfid(config_args, device)
        config_args["data"]["num_classes"] = temp

    def forward(self, x):
        pred = self.pred_network(x)
        _, uncertainty = self.uncertainty_network(x)
        return pred, uncertainty
