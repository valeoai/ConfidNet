import os

import numpy as np
import torch.utils.data as data
from PIL import Image

class_names = [
    "Sky",
    "Building",
    "Pole",
    # 'Road_marking',
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
    "Unlabelled",
]

# color codes for displaying segmentation results
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road_marking = [255, 69, 0]  # not used
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

label_colors = np.array(
    [
        Sky,
        Building,
        Pole,
        Road,
        Pavement,
        Tree,
        SignSymbol,
        Fence,
        Car,
        Pedestrian,
        Bicyclist,
        Unlabelled,
    ],
    dtype=np.uint8,
)

# fetched from Kendall
class_weights = np.array(
    [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 0.6823, 6.2478, 7.3614, 0.0]
)


class CamvidDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        transform=None,
        list_dir=None,
        img_size=[360, 480],
        crop_size=0,
        num_classes=12,
        phase=None,
        subset="",
    ):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.split = split
        self.phase = split if phase is None else phase
        self.img_size = img_size
        self.crop_size = crop_size
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.subset = subset
        self.num_classes = num_classes
        self.mean = np.array([0.411894, 0.425132, 0.432670])
        self.std = np.array([0.274135, 0.285062, 0.282846])
        self.read_lists()
        self.label_colors = label_colors
        self.class_names = class_names
        self.class_weights = class_weights
        self.ignore_label = 11
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(f"{self.data_dir}/{self.image_list[index]}")
        if self.label_list is not None:
            target = Image.open(f"{self.data_dir}/{self.label_list[index]}")

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, (target * 255).long()

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = self.data_dir / (self.split + ".txt")
        assert image_path.exists()
        self.image_list = [line.strip().split()[0] for line in open(image_path, "r")]
        self.label_list = [line.strip().split()[1] for line in open(image_path, "r")]
        print(f"fetched {len(self.image_list)} images from text file")
        assert len(self.image_list) == len(self.label_list)

    # input prediction should be a torch Tensor
    def decode_segmap(self, prediction, plot=False):
        return Image.fromarray(self.label_colors[prediction.numpy().squeeze()])

    # get raw image prior to normalization
    # expects input image as torch Tensor
    def unprocess_image(self, im, plot=False):
        im = im.squeeze().numpy().transpose((1, 2, 0))
        im = self.std * im + self.mean
        im = np.clip(im, 0, 1)
        im = im * 255
        im = Image.fromarray(im.astype(np.uint8))
        return im

    # de-center images and bring them back to their raw state
    # input: torch.tensor
    # output: torch.tensor
    def unprocess_batch(self, input):

        for i in range(input.size(1)):
            input[:, i, :, :] = self.std[i] * input[:, i, :, :]
            input[:, i, :, :] = input[:, i, :, :] + self.mean[i]
            input[:, i, :, :].clamp_(0, 1)

        return input
