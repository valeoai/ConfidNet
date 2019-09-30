import os
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

from confidnet.learners.learner import AbstractLeaner
from confidnet.utils import misc
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics

LOGGER = get_logger(__name__, level="DEBUG")


class OODConfidLearner(AbstractLeaner):

    def train(self, epoch):
        self.model.train()
        metrics = Metrics(
            self.metrics, self.prod_train_len, self.num_classes
        )
        loss, nll_loss, confid_loss = 0, 0, 0
        len_steps, len_data = 0, 0

        # Training loop
        loop = tqdm(self.train_loader)
        for batch_id, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.task == "classification":
                current_loss = self.criterion(output, target)
            elif self.task == "segmentation":
                current_loss = self.criterion(output, target.squeeze(dim=1))
            current_loss.backward()
            loss += current_loss
            nll_loss += self.criterion.loss_nll
            confid_loss += self.criterion.loss_confid
            self.optimizer.step()
            if self.task == "classification":
                len_steps += len(data)
                len_data = len_steps
            elif self.task == "segmentation":
                len_steps += len(data) * np.prod(data.shape[-2:])
                len_data += len(data)
            # Update metrics
            pred = output[0].argmax(dim=1, keepdim=True)
            confidence = torch.sigmoid(output[1])
            metrics.update(pred, target, confidence)

            # Update the average loss
            loop.set_description(f"Epoch {epoch}/{self.nb_epochs}")
            loop.set_postfix(
                OrderedDict(
                    {
                        "loss": f"{(loss / len_data):05.3e}",
                        "nll_loss": f"{(nll_loss / len_data):05.3e}",
                        "confid_loss": f"{(confid_loss / len_data):05.3e}",
                        "acc": f"{(metrics.accuracy / len_steps):05.2%}",
                    }
                )
            )
            loop.update()

        # Eval on epoch end
        scores = metrics.get_scores(split="train")
        logs_dict = OrderedDict(
            {
                "epoch": {"value": epoch, "string": f"{epoch:03}"},
                "train/loss": {
                    "value": loss / len_data,
                    "string": f"{(loss / len_data):05.4e}",
                },
                "train/loss_nll": {
                    "value": nll_loss / len_data,
                    "string": f"{(nll_loss / len_data):05.4e}",
                },
                "train/loss_confid": {
                    "value": confid_loss / len_data,
                    "string": f"{(confid_loss / len_data):05.4e}",
                },
            }
        )
        for s in scores:
            logs_dict[s] = scores[s]

        # Val scores
        val_losses, scores_val = self.evaluate(self.val_loader, self.prod_val_len, split="val")
        logs_dict["val/loss"] = {
            "value": val_losses["loss"].item() / self.nsamples_val,
            "string": f"{(val_losses['loss'].item() / self.nsamples_val):05.4e}",
        }
        logs_dict["val/loss_nll"] = {
            "value": val_losses["loss_nll"].item() / self.nsamples_val,
            "string": f"{(val_losses['loss_nll'].item() / self.nsamples_val):05.4e}",
        }
        logs_dict["val/loss_confid"] = {
            "value": val_losses["loss_confid"].item() / self.nsamples_val,
            "string": f"{(val_losses['loss_confid'].item() / self.nsamples_val):05.4e}",
        }
        for sv in scores_val:
            logs_dict[sv] = scores_val[sv]

        # Test scores
        test_losses, scores_test = self.evaluate(self.test_loader, self.prod_test_len, split="test")
        logs_dict["test/loss"] = {
            "value": test_losses["loss"].item() / self.nsamples_test,
            "string": f"{(test_losses['loss'].item() / self.nsamples_test):05.4e}",
        }
        logs_dict["test/loss_nll"] = {
            "value": test_losses["loss_nll"].item() / self.nsamples_test,
            "string": f"{(test_losses['loss_nll'].item() / self.nsamples_test):05.4e}",
        }
        logs_dict["test/loss_confid"] = {
            "value": test_losses["loss_confid"].item() / self.nsamples_test,
            "string": f"{(test_losses['loss_confid'].item() / self.nsamples_test):05.4e}",
        }
        for st in scores_test:
            logs_dict[st] = scores_test[st]

        # Print metrics
        misc.print_dict(logs_dict)

        # Save the model checkpoint
        self.save_checkpoint(epoch)

        # CSV logging
        misc.csv_writter(path=self.output_folder / "logs.csv", dic=OrderedDict(logs_dict))

        # Tensorboard logging
        self.save_tb(logs_dict)

        # Scheduler step
        if self.scheduler:
            self.scheduler.step()

    def evaluate(self, dloader, len_dataset, split="test", verbose=False, **args):
        self.model.eval()
        metrics = Metrics(self.metrics, len_dataset, self.num_classes)
        loss, nll_loss, confid_loss = 0, 0, 0

        # Evaluation loop
        loop = tqdm(dloader, disable=not verbose)
        for batch_id, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                output = self.model(data)
                if self.task == "classification":
                    loss += self.criterion(output, target)
                elif self.task == "segmentation":
                    loss += self.criterion(output, target.squeeze(dim=1))
                nll_loss += self.criterion.loss_nll
                confid_loss += self.criterion.loss_confid
                # Update metrics
                pred = output[0].argmax(dim=1, keepdim=True)
                confidence = torch.sigmoid(output[1])
                metrics.update(pred, target, confidence)

        scores = metrics.get_scores(split=split)
        losses = {"loss": loss, "loss_nll": nll_loss, "loss_confid": confid_loss}
        return losses, scores
