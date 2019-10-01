import csv
import os
from pathlib import Path

import torch
import yaml


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def csv_writter(path, dic):
    # Check if the file already exists
    if path.is_file():
        append_mode = True
        rw_mode = "a"
    else:
        append_mode = False
        rw_mode = "w"

    # Write dic
    with open(path, rw_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        # Do not write header in append mode
        if append_mode is False:
            writer.writerow(dic.keys())
        writer.writerow([elem["string"] for elem in dic.values()])


def print_dict(logs_dict):
    str_print = ""
    for metr_name in logs_dict:
        str_print += f"{metr_name}={logs_dict[metr_name]['string']},  "
    print(str_print)


def load_yaml(path):
    with open(path, "r") as f:
        config_args = yaml.load(f, Loader=yaml.SafeLoader)

    config_args["data"]["data_dir"] = Path(config_args["data"]["data_dir"])
    config_args["training"]["output_folder"] = Path(config_args["training"]["output_folder"])
    if config_args["model"]["resume"] not in [None, "vgg16"]:
        config_args["model"]["resume"] = Path(config_args["model"]["resume"])
    return config_args
