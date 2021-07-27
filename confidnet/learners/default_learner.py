from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from confidnet.learners.learner import AbstractLeaner
from confidnet.utils import misc
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics

LOGGER = get_logger(__name__, level="DEBUG")


class DefaultLearner(AbstractLeaner):

    def train(self, epoch):
        self.model.train()
        metrics = Metrics(
            self.metrics, self.prod_train_len, self.num_classes
        )
        loss, len_steps, len_data = 0, 0, 0

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
            self.optimizer.step()
            if self.task == "classification":
                len_steps += len(data)
                len_data = len_steps
            elif self.task == "segmentation":
                len_steps += len(data) * np.prod(data.shape[-2:])
                len_data += len(data)

            # Update metrics
            confidence, pred = F.softmax(output, dim=1).max(dim=1, keepdim=True)
            metrics.update(pred, target, confidence)

            # Update the average loss
            loop.set_description(f"Epoch {epoch}/{self.nb_epochs}")
            loop.set_postfix(
                OrderedDict(
                    {
                        "loss_nll": f"{(loss / len_data):05.4e}",
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
                "lr": {
                    "value": self.optimizer.param_groups[0]["lr"],
                    "string": f"{self.optimizer.param_groups[0]['lr']:05.1e}",
                },
                "train/loss_nll": {
                    "value": loss / len_data,
                    "string": f"{(loss / len_data):05.4e}",
                },
            }
        )
        for s in scores:
            logs_dict[s] = scores[s]

        # Val scores
        if self.val_loader is not None:
            val_losses, scores_val = self.evaluate(self.val_loader, self.prod_val_len, split="val")
            logs_dict["val/loss_nll"] = {
                "value": val_losses["loss_nll"].item() / self.nsamples_val,
                "string": f"{(val_losses['loss_nll'].item() / self.nsamples_val):05.4e}",
            }
            for sv in scores_val:
                logs_dict[sv] = scores_val[sv]

        # Test scores
        test_losses, scores_test = self.evaluate(self.test_loader, self.prod_test_len, split="test")
        logs_dict["test/loss_nll"] = {
            "value": test_losses["loss_nll"].item() / self.nsamples_test,
            "string": f"{(test_losses['loss_nll'].item() / self.nsamples_test):05.4e}",
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

    def evaluate(
        self, dloader, len_dataset, split="test", mode="mcp", samples=50, verbose=False
    ):
        self.model.eval()
        metrics = Metrics(self.metrics, len_dataset, self.num_classes)
        loss = 0

        # Special case of mc-dropout
        if mode == "mc_dropout":
            self.model.keep_dropout_in_test()
            LOGGER.info(f"Sampling {samples} times")

        # Evaluation loop
        loop = tqdm(dloader, disable=not verbose)
        for batch_id, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                if mode == "mcp":
                    output = self.model(data)
                    if self.task == "classification":
                        loss += self.criterion(output, target)
                    elif self.task == "segmentation":
                        loss += self.criterion(output, target.squeeze(dim=1))
                    confidence, pred = F.softmax(output, dim=1).max(dim=1, keepdim=True)

                elif mode == "tcp":
                    output = self.model(data)
                    if self.task == "classification":
                        loss += self.criterion(output, target)
                    elif self.task == "segmentation":
                        loss += self.criterion(output, target.squeeze(dim=1))
                    probs = F.softmax(output, dim=1)
                    pred = probs.max(dim=1, keepdim=True)[1]
                    labels_hot = misc.one_hot_embedding(
                        target, self.num_classes
                    ).to(self.device)
                    # Segmentation special case
                    if self.task == "segmentation":
                        labels_hot = labels_hot.squeeze(1).permute(0, 3, 1, 2)
                    confidence, _ = (labels_hot * probs).max(dim=1, keepdim=True)

                elif mode == "mc_dropout":
                    if self.task == "classification":
                        outputs = torch.zeros(
                            samples, data.shape[0], self.num_classes
                        ).to(self.device)
                    elif self.task == "segmentation":
                        outputs = torch.zeros(
                            samples,
                            data.shape[0],
                            self.num_classes,
                            data.shape[2],
                            data.shape[3],
                        ).to(self.device)
                    for i in range(samples):
                        outputs[i] = self.model(data)
                    output = outputs.mean(0)
                    if self.task == "classification":
                        loss += self.criterion(output, target)
                    elif self.task == "segmentation":
                        loss += self.criterion(output, target.squeeze(dim=1))
                    probs = F.softmax(output, dim=1)
                    confidence = (probs * torch.log(probs + 1e-9)).sum(dim=1)  # entropy
                    pred = probs.max(dim=1, keepdim=True)[1]

                metrics.update(pred, target, confidence)

        scores = metrics.get_scores(split=split)
        losses = {"loss_nll": loss}
        return losses, scores
