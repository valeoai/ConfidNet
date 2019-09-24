from models.model import AbstractModel
from models.segnet import Segnet
from models.segnet_selfconfid import SegnetSelfConfid

class SegnetSelfConfidCloning(AbstractModel):
    def __init__(self, config_args, device):
        super(SegnetSelfConfidCloning, self).__init__(config_args, device)
        self.pred_network = Segnet(config_args, device)

        # Small trick to set num classes to 1
        temp = config_args['data']['num_classes']
        config_args['data']['num_classes'] = 1
        self.uncertainty_network = SegnetSelfConfid(config_args, device)
        config_args['data']['num_classes'] = temp

    def forward(self, x):
        pred = self.pred_network(x)
        _, uncertainty = self.uncertainty_network(x)
        return pred, uncertainty

    def print_summary(self, input_size):
        pass
