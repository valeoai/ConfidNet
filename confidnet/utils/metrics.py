import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist


class Metrics:
    def __init__(self, metrics, len_dataset, n_classes):
        self.metrics = metrics
        self.len_dataset = len_dataset
        self.n_classes = n_classes
        self.accurate, self.errors, self.proba_pred = [], [], []
        self.accuracy = 0
        self.current_miou = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, pred, target, confidence):
        self.accurate.extend(pred.eq(target.view_as(pred)).detach().to('cpu').numpy())
        self.accuracy += pred.eq(target.view_as(pred)).sum().item()
        self.errors.extend((pred != target.view_as(pred)).detach().to('cpu').numpy())
        self.proba_pred.extend(confidence.detach().to('cpu').numpy())

        if 'mean_iou' in self.metrics:
            pred = pred.cpu().numpy().flatten()
            target = target.cpu().numpy().flatten()
            mask = (target >= 0) & (target < self.n_classes)
            hist = np.bincount(
                self.n_classes * target[mask].astype(int) + pred[mask], minlength=self.n_classes ** 2
            ).reshape(self.n_classes, self.n_classes)
            self.confusion_matrix += hist

    def get_scores(self, split='train'):
        self.accurate = np.reshape(self.accurate, newshape=(len(self.accurate), -1)).flatten()
        self.errors = np.reshape(self.errors, newshape=(len(self.errors), -1)).flatten()
        self.proba_pred = np.reshape(self.proba_pred, newshape=(len(self.proba_pred), -1)).flatten()

        scores = {}
        if 'accuracy' in self.metrics:
            accuracy = self.accuracy / float(self.len_dataset)
            scores[f'{split}/accuracy'] = {'value':accuracy, 'string':f'{accuracy:05.2%}'}
        if 'auc' in self.metrics:
            if len(np.unique(self.accurate)) == 1:
                auc = 1
            else:
                auc = roc_auc_score(self.accurate, self.proba_pred)
            scores[f'{split}/auc'] = {'value':auc, 'string':f'{auc:05.2%}'}
        if 'ap_success' in self.metrics:
            ap_success = average_precision_score(self.accurate, self.proba_pred)
            scores[f'{split}/ap_success'] = {'value':ap_success, 'string':f'{ap_success:05.2%}'}
        if 'accuracy_success' in self.metrics:
            accuracy_success = np.round(self.proba_pred[self.accurate==1]).mean()
            scores[f'{split}/accuracy_success'] = {'value':accuracy_success, 'string':f'{accuracy_success:05.2%}'}
        if 'ap_errors' in self.metrics:
            ap_errors = average_precision_score(self.errors, -self.proba_pred)
            scores[f'{split}/ap_errors'] = {'value':ap_errors, 'string':f'{ap_errors:05.2%}'}
        if 'accuracy_errors' in self.metrics:
            accuracy_errors = 1. - np.round(self.proba_pred[self.errors==1]).mean()
            scores[f'{split}/accuracy_errors'] = {'value':accuracy_errors, 'string':f'{accuracy_errors:05.2%}'}
        if 'fpr_at_95tpr' in self.metrics:
            for delta in np.arange(self.proba_pred.min(), self.proba_pred.max(), (self.proba_pred.max()-self.proba_pred.min())/10000):
                tpr = len(self.proba_pred[(self.accurate==1) & (self.proba_pred>=delta)]) / float(len(self.proba_pred[(self.accurate==1)]))
                print(delta)
                print(tpr)
                print('------')
                if 0.9505 >= tpr >= 0.9495:
                    print(f'Nearest threshold 95% TPR value: {tpr}')
                    print(f'Threshold 95% TPR value: {delta}')
                    fpr = len(self.proba_pred[(self.errors==1) & (self.proba_pred>=delta)]) / float(len(self.proba_pred[(self.errors==1)]))
                    scores[f'{split}/fpr_at_95tpr'] = {'value': fpr, 'string': f'{fpr:05.2%}'}
                    break
        if 'mean_iou' in self.metrics:
            iou = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix))
            mean_iou = np.nanmean(iou)
            scores[f'{split}/mean_iou'] = {'value': mean_iou, 'string': f'{mean_iou:05.2%}'}

        return scores
