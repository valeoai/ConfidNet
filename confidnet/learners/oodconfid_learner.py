import os
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict
from confidnet.learners.learner import AbstractLeaner
from confidnet.utils.metrics import Metrics
from confidnet.utils import misc
from confidnet.utils.logger import get_logger
LOGGER = get_logger(__name__, level='DEBUG')

class OODConfidLearner(AbstractLeaner):
    def __init__(self, config_args, train_loader, val_loader, test_loader, start_epoch, device):
        super(OODConfidLearner, self).__init__(config_args, train_loader, val_loader, test_loader, start_epoch, device)

    def train(self, epoch):
        self.model.train()
        metrics = Metrics(self.metrics, self.prod_train_len, self.config_args['data']['num_classes'])
        loss, nll_loss, confid_loss = 0, 0, 0
        len_steps, len_data = 0, 0

        # Training loop
        loop = tqdm(self.train_loader)
        for batch_id, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.task == 'classification':
                current_loss = self.criterion(output, target)
            elif self.task == 'segmentation':
                current_loss = self.criterion(output, target.squeeze(dim=1))
            current_loss.backward()
            loss += current_loss
            nll_loss += self.criterion.loss_nll
            confid_loss += self.criterion.loss_confid
            self.optimizer.step()
            if self.task == 'classification':
                len_steps += len(data)
                len_data = len_steps
            elif self.task == 'segmentation':
                len_steps += len(data) * np.prod(data.shape[-2:])
                len_data += len(data)
            # Update metrics
            pred = output[0].argmax(dim=1, keepdim=True)
            confidence = torch.sigmoid(output[1])
            metrics.update(pred, target, confidence)

            # Update the average loss
            loop.set_description('Epoch {}/{}'.format(epoch, self.nb_epochs))
            loop.set_postfix(OrderedDict({'loss':'{:05.3e}'.format(loss / float(len_data)),
                'nll_loss':'{:05.3e}'.format(nll_loss / float(len_data)),
                'confid_loss':'{:05.3e}'.format(confid_loss / float(len_data)),
                'acc':'{:05.2%}'.format(metrics.accuracy / float(len_steps))}))
            loop.update()

        # Eval on epoch end
        scores = metrics.get_scores(split='train')
        logs_dict = OrderedDict({'epoch': {'value': epoch,
                                           'string': '{:03}'.format(epoch)},
                                 'train/loss': {'value': loss / float(len_data),
                                                'string': '{:05.4e}'.format(loss / float(len_data))},
                                 'train/loss_nll': {'value': nll_loss / float(len_data),
                                                    'string': '{:05.4e}'.format(nll_loss / float(len_data))},
                                 'train/loss_confid': {'value': confid_loss / float(len_data),
                                                       'string': '{:05.4e}'.format(confid_loss / float(len_data))},
                                 })
        for s in scores:
            logs_dict[s] = scores[s]

        # Val scores
        val_losses, scores_val = self.evaluate(self.val_loader, self.prod_val_len, split='val')
        logs_dict['val/loss'] = {'value': val_losses['loss'].item() / float(self.nsamples_val),
                                 'string': '{:05.4e}'.format(val_losses['loss'].item() / float(self.nsamples_val))}
        logs_dict['val/loss_nll'] = {'value': val_losses['loss_nll'].item() / float(self.nsamples_val),
                                     'string': '{:05.4e}'.format(
                                         val_losses['loss_nll'].item() / float(self.nsamples_val))}
        logs_dict['val/loss_confid'] = {'value': val_losses['loss_confid'].item() / float(self.nsamples_val),
                                        'string': '{:05.4e}'.format(
                                            val_losses['loss_confid'].item() / float(self.nsamples_val))}
        for sv in scores_val:
            logs_dict[sv] = scores_val[sv]

        # Test scores
        test_losses, scores_test = self.evaluate(self.test_loader, self.prod_test_len, split='test')
        logs_dict['test/loss'] = {'value': test_losses['loss'].item() / float(self.nsamples_test),
                                  'string': '{:05.4e}'.format(
                                      test_losses['loss'].item() / float(self.nsamples_test))}
        logs_dict['test/loss_nll'] = {'value': test_losses['loss_nll'].item() / float(self.nsamples_test),
                                      'string': '{:05.4e}'.format(
                                          test_losses['loss_nll'].item() / float(self.nsamples_test))}
        logs_dict['test/loss_confid'] = {'value': test_losses['loss_confid'].item() / float(self.nsamples_test),
                                         'string': '{:05.4e}'.format(
                                             test_losses['loss_confid'].item() / float(self.nsamples_test))}
        for st in scores_test:
            logs_dict[st] = scores_test[st]

        # Print metrics
        misc.print_dict(logs_dict)

        # Save the model checkpoint
        self.save_checkpoint(epoch)

        # CSV logging
        misc.csv_writter(path=os.path.join(self.output_folder, 'logs.csv'), dic=OrderedDict(logs_dict))

        # Tensorboard logging
        self.save_tb(logs_dict)

        # Scheduler step
        if self.scheduler:
            self.scheduler.step()

    def evaluate(self, dloader, len_dataset, split='test', verbose=False, **args):
        self.model.eval()
        metrics = Metrics(self.metrics, len_dataset, self.config_args['data']['num_classes'])
        loss, nll_loss, confid_loss = 0, 0, 0

        # Evaluation loop
        loop = tqdm(dloader, disable=not verbose)
        for batch_id, (data, target) in enumerate(loop):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                output = self.model(data)
                if self.task == 'classification':
                    loss += self.criterion(output, target)
                elif self.task == 'segmentation':
                    loss += self.criterion(output, target.squeeze(dim=1))
                nll_loss += self.criterion.loss_nll
                confid_loss += self.criterion.loss_confid
                # Update metrics
                pred = output[0].argmax(dim=1, keepdim=True)
                confidence = torch.sigmoid(output[1])
                metrics.update(pred, target, confidence)

        scores = metrics.get_scores(split=split)
        losses = {'loss': loss, 'loss_nll': nll_loss, 'loss_confid': confid_loss}
        return losses, scores
