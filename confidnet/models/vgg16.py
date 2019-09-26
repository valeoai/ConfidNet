import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from confidnet.models.model import AbstractModel


class Conv2dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class VGG16(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.conv1 = Conv2dSame(config_args["data"]["input_channels"], 64, 3)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_dropout = nn.Dropout(0.3)
        self.conv2 = Conv2dSame(64, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = Conv2dSame(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_dropout = nn.Dropout(0.4)
        self.conv4 = Conv2dSame(128, 128, 3)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = Conv2dSame(128, 256, 3)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv5_dropout = nn.Dropout(0.4)
        self.conv6 = Conv2dSame(256, 256, 3)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv6_dropout = nn.Dropout(0.4)
        self.conv7 = Conv2dSame(256, 256, 3)
        self.conv7_bn = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv8 = Conv2dSame(256, 512, 3)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.conv8_dropout = nn.Dropout(0.4)
        self.conv9 = Conv2dSame(512, 512, 3)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv9_dropout = nn.Dropout(0.4)
        self.conv10 = Conv2dSame(512, 512, 3)
        self.conv10_bn = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv11 = Conv2dSame(512, 512, 3)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv11_dropout = nn.Dropout(0.4)
        self.conv12 = Conv2dSame(512, 512, 3)
        self.conv12_bn = nn.BatchNorm2d(512)
        self.conv12_dropout = nn.Dropout(0.4)
        self.conv13 = Conv2dSame(512, 512, 3)
        self.conv13_bn = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(2)

        self.end_dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512, 512)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, config_args["data"]["num_classes"])

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv1_bn(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.3, training=self.training)
        else:
            out = self.conv1_dropout(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = self.maxpool1(out)

        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.4, training=self.training)
        else:
            out = self.conv3_dropout(out)
        out = F.relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = self.maxpool2(out)

        out = F.relu(self.conv5(out))
        out = self.conv5_bn(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.4, training=self.training)
        else:
            out = self.conv5_dropout(out)
        out = F.relu(self.conv6(out))
        out = self.conv6_bn(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.4, training=self.training)
        else:
            out = self.conv6_dropout(out)
        out = F.relu(self.conv7(out))
        out = self.conv7_bn(out)
        out = self.maxpool3(out)

        out = F.relu(self.conv8(out))
        out = self.conv8_bn(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.4, training=self.training)
        else:
            out = self.conv8_dropout(out)
        out = F.relu(self.conv9(out))
        out = self.conv9_bn(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.4, training=self.training)
        else:
            out = self.conv9_dropout(out)
        out = F.relu(self.conv10(out))
        out = self.conv10_bn(out)
        out = self.maxpool4(out)

        out = F.relu(self.conv11(out))
        out = self.conv11_bn(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.4, training=self.training)
        else:
            out = self.conv11_dropout(out)
        out = F.relu(self.conv12(out))
        out = self.conv12_bn(out)
        if self.mc_dropout:
            out = F.dropout(out, 0.4, training=self.training)
        else:
            out = self.conv12_dropout(out)
        out = F.relu(self.conv13(out))
        out = self.conv13_bn(out)
        out = self.maxpool5(out)

        if self.mc_dropout:
            out = F.dropout(out, 0.5, training=self.training)
        else:
            out = self.end_dropout(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.mc_dropout:
            out = F.dropout(out, 0.5, training=self.training)
        else:
            out = self.dropout_fc(out)
        out = self.fc2(out)
        return out

    def init_vgg16_params(self):
        vgg16 = models.vgg16(pretrained=True).to(self.device)
        vgg_layers = []
        for _layer in vgg16.features.children():
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        model_layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
            self.conv7,
            self.conv8,
            self.conv9,
            self.conv10,
            self.conv11,
            self.conv12,
            self.conv13,
        ]

        assert len(vgg_layers) == len(model_layers)

        for l1, l2 in zip(vgg_layers, model_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
