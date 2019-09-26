from torchvision import transforms

from confidnet.augmentations import segmentation_transforms
from confidnet.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")

CLASSIFICATION_AUGMENTATION = {
    "compose": lambda x: transforms.Compose(x),
    "to_tensor": transforms.ToTensor,
    "normalize": lambda x: transforms.Normalize(x[0], x[1]),
    "random_crop": lambda x: transforms.RandomCrop(x, padding=4),
    "center_crop": lambda x: transforms.CenterCrop(x),
    "hflip": lambda x: transforms.RandomHorizontalFlip() if x else None,
    "resize": lambda x: transforms.Resize(x),
    "rotate": lambda x: transforms.RandomRotation(x),
    "color_jitter": lambda x: transforms.ColorJitter(
        brightness=x[0], contrast=x[1], saturation=x[2], hue=x[3]
    ),
}

SEGMENTATION_AUGMENTATION = {
    "compose": lambda x: segmentation_transforms.Compose(x),
    "to_tensor": segmentation_transforms.ToTensor,
    "normalize": lambda x: segmentation_transforms.Normalize(x[0], x[1]),
    "random_crop": lambda x: segmentation_transforms.RandomCrop(x, padding=4),
    "center_crop": lambda x: segmentation_transforms.CenterCrop(x),
    "hflip": lambda x: segmentation_transforms.RandomHorizontallyFlip(p=0.5) if x else None,
    "resize": lambda x: segmentation_transforms.Scale(x),
    "rotate": lambda x: segmentation_transforms.RandomRotate(x),
    "color_jitter": lambda x: transforms.ColorJitter(
        brightness=x[0], contrast=x[1], saturation=x[2], hue=x[3]
    ),
}


def get_composed_augmentations(aug_dict, verbose=True, training="classif"):
    # Switch between classif and segmentation
    if training == "classif":
        aug_type = CLASSIFICATION_AUGMENTATION
    elif training == "segmentation":
        aug_type = SEGMENTATION_AUGMENTATION
    else:
        raise KeyError(f"Augmentation dict {training} non existing")

    if aug_dict is None:
        LOGGER.info("Using No Augmentations")
        return aug_type["compose"]([aug_type["to_tensor"]()])

    augmentations, aug_after = [], []
    for aug_key, aug_param in aug_dict.items():
        if aug_key == "normalize":
            aug_after.append(aug_type[aug_key](aug_param))
            continue
        augmentations.append(aug_type[aug_key](aug_param))
        if verbose:
            LOGGER.info(f"Using {aug_key} aug with params {aug_param}")
    augmentations.append(aug_type["to_tensor"]())
    augmentations.extend(aug_after)
    return aug_type["compose"](augmentations)
