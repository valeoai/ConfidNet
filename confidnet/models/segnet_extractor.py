import torch.nn as nn
import torch.nn.functional as F

from confidnet.models.model import AbstractModel
from confidnet.models.segnet import segnetDown2, segnetDown3, segnetUp2, segnetUp3


class SegnetExtractor(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.in_channels = config_args["data"]["input_channels"]
        self.n_classes = config_args["data"]["num_classes"]
        self.is_unpooling = True
        self.dropout = config_args["model"]["is_dropout"]

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.dropout_down3 = nn.Dropout(0.5)
        self.down4 = segnetDown3(256, 512)
        self.dropout_down4 = nn.Dropout(0.5)
        self.down5 = segnetDown3(512, 512)
        self.dropout_down5 = nn.Dropout(0.5)

        self.up5 = segnetUp3(512, 512)
        self.dropout_up5 = nn.Dropout(0.5)
        self.up4 = segnetUp3(512, 256)
        self.dropout_up4 = nn.Dropout(0.4)
        self.up3 = segnetUp3(256, 128)
        self.dropout_up3 = nn.Dropout(0.3)
        self.up2 = segnetUp2(128, 64)
        self.last_unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        if self.dropout:
            if self.mc_dropout:
                down3 = F.dropout(down3, 0.5, training=self.training)
            else:
                down3 = self.dropout_down3(down3)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        if self.dropout:
            if self.mc_dropout:
                down4 = F.dropout(down4, 0.5, training=self.training)
            else:
                down4 = self.dropout_down3(down4)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        if self.dropout:
            if self.mc_dropout:
                down5 = F.dropout(down5, 0.5, training=self.training)
            else:
                down5 = self.dropout_down3(down5)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        if self.dropout:
            if self.mc_dropout:
                up5 = F.dropout(up5, 0.5, training=self.training)
            else:
                up5 = self.dropout_up5(up5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        if self.dropout:
            if self.mc_dropout:
                up4 = F.dropout(up4, 0.5, training=self.training)
            else:
                up4 = self.dropout_up4(up4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        if self.dropout:
            if self.mc_dropout:
                up3 = F.dropout(up3, 0.5, training=self.training)
            else:
                up3 = self.dropout_up3(up3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up2 = self.last_unpool(up2, indices_1, unpool_shape1)
        return up2

    def print_summary(self, input_size):
        pass
