import torch.nn as nn
from torchsummary import summary

from confidnet.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")


class AbstractModel(nn.Module):
    def __init__(self, config_args, device):
        super().__init__()
        self.device = device
        self.mc_dropout = config_args["training"].get("mc_dropout", None)

    def forward(self, x):
        pass

    def keep_dropout_in_test(self):
        if self.mc_dropout:
            LOGGER.warning("Keeping dropout activated during evaluation mode")
            self.training = True

    def print_summary(self, input_size):
        summary(self, input_size)
