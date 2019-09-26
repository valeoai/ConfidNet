import os

import torch
import torch.optim as optim

from confidnet.models import get_model
from confidnet.utils import losses
from confidnet.utils.logger import get_logger
from confidnet.utils.schedulers import get_scheduler

LOGGER = get_logger(__name__, level="DEBUG")


class AbstractLeaner:
    def __init__(self, config_args, train_loader, val_loader, test_loader, start_epoch, device):
        self.config_args = config_args
        self.num_classes = config_args['data']['num_classes']
        self.task = config_args["training"]["task"]
        self.loss_args = config_args["training"]["loss"]
        self.metrics = config_args["training"]["metrics"]
        self.nb_epochs = config_args["training"]["nb_epochs"]
        self.output_folder = config_args["training"]["output_folder"]
        self.lr_schedule = config_args["training"]["lr_schedule"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # Usually val set is made from train set, else compute len as usual
        try:
            self.nsamples_train = len(self.train_loader.sampler.indices)
            self.nsamples_val = len(self.val_loader.sampler.indices)
        except:
            self.nsamples_train = len(self.train_loader.dataset)
            self.nsamples_val = len(self.val_loader.dataset)
        self.nsamples_test = len(self.test_loader.dataset)
        # Segmentation case
        if self.task == "classification":
            self.prod_train_len = self.nsamples_train
            self.prod_val_len = self.nsamples_val
            self.prod_test_len = self.nsamples_test
        if self.task == "segmentation":
            self.prod_train_len = (
                self.nsamples_train
                * self.train_loader.dataset[0][0].shape[1]
                * self.train_loader.dataset[0][0].shape[2]
            )
            self.prod_val_len = (
                self.nsamples_val
                * self.val_loader.dataset[0][0].shape[1]
                * self.val_loader.dataset[0][0].shape[2]
            )
            self.prod_test_len = (
                self.nsamples_test
                * self.test_loader.dataset[0][0].shape[1]
                * self.test_loader.dataset[0][0].shape[2]
            )
        self.device = device
        self.last_epoch = start_epoch - 2
        self.criterion, self.scheduler, self.optimizer, self.tb_logger = None, None, None, None

        # Initialize model
        self.model = get_model(config_args, self.device).to(self.device)
        # Set optimizer
        self.set_optimizer(config_args["training"]["optimizer"]["name"])
        # Set loss
        self.set_loss()
        # Temperature scaling
        self.temperature = config_args["training"].get("temperature", None)

    def train(self, epoch):
        pass

    def set_loss(self):
        if self.loss_args["name"] in losses.CUSTOM_LOSS:
            self.criterion = losses.CUSTOM_LOSS[self.loss_args["name"]](
                config_args=self.config_args, device=self.device
            )
        elif self.loss_args["name"] in losses.PYTORCH_LOSS:
            self.criterion = losses.PYTORCH_LOSS[self.loss_args["name"]](ignore_index=255)
        else:
            raise Exception(f"Loss {self.loss_args['name']} not implemented")
        LOGGER.info(f"Using loss {self.loss_args['name']}")

    def set_optimizer(self, optimizer_name):
        optimizer_params = {
            k: v for k, v in self.config_args["training"]["optimizer"].items() if k != "name"
        }
        LOGGER.info(f"Using optimizer {optimizer_name}")
        if optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_params)
        elif optimizer_name == "adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters(), **optimizer_params)
        else:
            raise KeyError("Bad optimizer name or not implemented (sgd, adam, adadelta).")

    def set_scheduler(self):
        self.scheduler = get_scheduler(self.optimizer, self.lr_schedule, self.last_epoch)

    def load_checkpoint(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

    def save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict()
                if isinstance(self.model, torch.nn.DataParallel)
                else self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.output_folder / f"model_epoch_{epoch:03d}.ckpt",
        )

    def save_tb(self, logs_dict):
        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        epoch = logs_dict["epoch"]["value"]
        del logs_dict["epoch"]

        for tag in logs_dict:
            self.tb_logger.scalar_summary(tag, logs_dict[tag]["value"], epoch)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.model.named_parameters():
            tag = tag.replace(".", "/")
            self.tb_logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            if not value.grad is None:
                self.tb_logger.histo_summary(tag + "/grad", value.grad.data.cpu().numpy(), epoch)
