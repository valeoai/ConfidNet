import torch
from torchvision import datasets

from confidnet.augmentations import get_composed_augmentations
from confidnet.loaders.camvid_dataset import CamvidDataset
from confidnet.loaders.loader import AbstractDataLoader


class MNISTLoader(AbstractDataLoader):
    def load_dataset(self):
        self.train_dataset = datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        self.test_dataset = datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self.augmentations_test
        )


class SVHNLoader(AbstractDataLoader):
    def load_dataset(self):
        self.train_dataset = datasets.SVHN(
            root=self.data_dir, split="train", download=True, transform=self.augmentations_train
        )
        self.test_dataset = datasets.SVHN(
            root=self.data_dir, split="test", download=True, transform=self.augmentations_test
        )


class CIFAR10Loader(AbstractDataLoader):
    def load_dataset(self):
        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.augmentations_test
        )


class CIFAR100Loader(AbstractDataLoader):
    def load_dataset(self):
        self.train_dataset = datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=self.augmentations_test
        )


class CamVidLoader(AbstractDataLoader):
    def add_augmentations(self):
        self.augmentations_train = get_composed_augmentations(
            self.augmentations, training="segmentation"
        )
        self.augmentations_val = get_composed_augmentations(
            {
                key: self.augmentations[key]
                for key in self.augmentations
                if key in ["normalize", "resize"]
            },
            verbose=False,
            training="segmentation",
        )
        self.augmentations_test = get_composed_augmentations(
            {key: self.augmentations[key] for key in self.augmentations if key == "normalize"},
            verbose=False,
            training="segmentation",
        )

    def load_dataset(self):
        # Loading dataset
        self.train_dataset = CamvidDataset(
            data_dir=self.data_dir, split="train", transform=self.augmentations_train
        )
        self.val_dataset = CamvidDataset(
            data_dir=self.data_dir, split="val", transform=self.augmentations_val
        )
        self.test_dataset = CamvidDataset(
            data_dir=self.data_dir, split="test", transform=self.augmentations_test
        )

    def make_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
