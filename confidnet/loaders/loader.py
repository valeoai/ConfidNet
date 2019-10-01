from pathlib import Path
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from confidnet.augmentations import get_composed_augmentations
from confidnet.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")


class AbstractDataLoader:    
    def __init__(self, config_args):
        self.output_folder = config_args['training']['output_folder']
        self.data_dir = config_args['data']['data_dir']
        self.batch_size = config_args['training']['batch_size']
        self.img_size = (config_args['data']['input_size'][0],
                         config_args['data']['input_size'][1],
                         config_args['data']['input_channels'])
        self.augmentations = config_args['training'].get('augmentations', None)
        self.ft_on_val = config_args['training'].get('ft_on_val', None)
        self.resume_folder = config_args['model']['resume'].parent if isinstance(config_args['model']['resume'], Path) else None
        self.valid_size = config_args['data']['valid_size']
        self.perturbed_folder = config_args['data'].get('perturbed_images', None)
        self.pin_memory = config_args['training']['pin_memory']
        self.num_workers = config_args['training']['num_workers']
        self.train_loader, self.val_loader, self.test_loader = None, None, None

        # Set up augmentations
        self.augmentations_train, self.augmentations_train_lbl = None, None
        self.augmentations_test, self.augmentations_test_lbl = None, None
        if self.augmentations:
            LOGGER.info("--- Augmentations ---")
            self.add_augmentations()

        # Load dataset
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.load_dataset()

    def add_augmentations(self):
        self.augmentations_train = get_composed_augmentations(
            self.augmentations, training="classif"
        )
        self.augmentations_train_lbl = get_composed_augmentations(
            {
                key: self.augmentations[key]
                for key in self.augmentations
                if key not in ["normalize", "color_jitter"]
            },
            verbose=False,
            training="classif",
        )
        self.augmentations_test = get_composed_augmentations(
            {key: self.augmentations[key] for key in self.augmentations if key == "normalize"},
            verbose=False,
            training="classif",
        )
        self.augmentations_test_lbl = get_composed_augmentations(
            None, verbose=False, training="classif"
        )

    def load_dataset(self):
        pass

    def make_loaders(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

        if self.valid_size == 0:
            LOGGER.warning("Valid size=0, no validation loader")
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
        else:
            num_train = len(self.train_dataset)
            indices = list(range(num_train))

            if (self.output_folder / "train_idx.npy").exists():
                LOGGER.warning("Loading existing train-val split indices")
                train_idx = np.load(self.output_folder / "train_idx.npy")
                val_idx = np.load(self.output_folder / "val_idx.npy")
            # Splitting indices
            elif self.resume_folder:
                LOGGER.warning("Loading existing train-val split indices from ORIGINAL training")
                train_idx = np.load(self.resume_folder / "train_idx.npy")
                val_idx = np.load(self.resume_folder / "val_idx.npy")
            else:
                split = int(np.floor(self.valid_size * num_train))
                np.random.seed(42)
                np.random.shuffle(indices)
                train_idx, val_idx = indices[split:], indices[:split]
                np.save(self.output_folder / "train_idx.npy", train_idx)
                np.save(self.output_folder / "val_idx.npy", val_idx)
            # Make samplers
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            # Special case where val set is used for training
            if self.ft_on_val:
                LOGGER.warning("Using val set as training")
                train_sampler = val_sampler
            # Make loaders
            self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
            self.val_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                sampler=val_sampler,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
