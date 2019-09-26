import structured_map_ranking_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from confidnet.utils import misc


class SelfConfidMSELoss(nn.modules.loss._Loss):
    def __init__(self, config_args, device):
        self.nb_classes = config_args["data"]["num_classes"]
        self.task = config_args["training"]["task"]
        self.weighting = config_args["training"]["loss"]["weighting"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input[0], dim=1)
        confidence = torch.sigmoid(input[1]).squeeze()
        # Apply optional weighting
        weights = torch.ones_like(target).type(torch.FloatTensor).to(self.device)
        weights[(probs.argmax(dim=1) != target)] *= self.weighting
        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(self.device)
        # Segmentation special case
        if self.task == "segmentation":
            labels_hot = labels_hot.permute(0, 3, 1, 2)
        loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2
        return torch.mean(loss)


class SelfConfidTCPRLoss(nn.modules.loss._Loss):
    def __init__(self, config_args, device):
        self.nb_classes = config_args["data"]["num_classes"]
        self.task = config_args["training"]["task"]
        self.weighting = config_args["training"]["loss"]["weighting"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input[0], dim=1)
        maxprob = probs.max(dim=1)[0]
        confidence = torch.sigmoid(input[1]).squeeze()
        # Apply optional weighting
        weights = torch.ones_like(target).type(torch.FloatTensor).to(self.device)
        weights[(probs.argmax(dim=1) != target)] *= self.weighting
        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(self.device)
        # Segmentation special case
        if self.task == "segmentation":
            labels_hot = labels_hot.permute(0, 3, 1, 2)
        loss = weights * (confidence - (probs * labels_hot).sum(dim=1) / maxprob) ** 2
        return torch.mean(loss)


class SelfConfidBCELoss(nn.modules.loss._Loss):
    def __init__(self, device, config_args):
        self.nb_classes = config_args["data"]["num_classes"]
        self.weighting = config_args["training"]["loss"]["weighting"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        confidence = input[1].squeeze(dim=1)
        weights = torch.ones_like(target).type(torch.FloatTensor).to(self.device)
        weights[(input[0].argmax(dim=1) != target)] *= self.weighting
        return nn.BCEWithLogitsLoss(weight=weights)(
            confidence, (input[0].argmax(dim=1) == target).float()
        )


class FocalLoss(nn.modules.loss._Loss):
    def __init__(self, config_args, device):
        super().__init__()
        self.alpha = config_args["training"]["loss"].get("alpha", 0.25)
        self.gamma = config_args["training"]["loss"].get("gamma", 5)

    def forward(self, input, target):
        confidence = input[1].squeeze(dim=1)
        BCE_loss = F.binary_cross_entropy_with_logits(
            confidence, (input[0].argmax(dim=1) == target).float(), reduction="none"
        )
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(loss)


class StructuredMAPRankingLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, mask):
        loss, ranking_lai = structured_map_ranking_loss.forward(input, target, mask)
        ctx.save_for_backward(input, target, mask, ranking_lai)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target, mask, ranking_lai = ctx.saved_variables
        grad_input = structured_map_ranking_loss.backward(
            grad_output, input, target, mask, ranking_lai
        )
        return grad_input, None, None


class StructuredMAPRankingLoss(nn.modules.loss._Loss):
    def __init__(self, device, config_args):
        self.nb_classes = config_args["data"]["num_classes"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        confidence = input[1]
        mask = torch.ones_like(target).unsqueeze(dim=1)
        return StructuredMAPRankingLossFunction.apply(
            confidence,
            (input[0].argmax(dim=1) == target).float().unsqueeze(dim=1),
            mask.to(dtype=torch.uint8),
        )


class OODConfidenceLoss(nn.modules.loss._Loss):
    def __init__(self, device, config_args):
        self.nb_classes = config_args["data"]["num_classes"]
        self.task = config_args["training"]["task"]
        self.device = device
        self.half_random = config_args["training"]["loss"]["half_random"]
        self.beta = config_args["training"]["loss"]["beta"]
        self.lbda = config_args["training"]["loss"]["lbda"]
        self.lbda_control = config_args["training"]["loss"]["lbda_control"]
        self.loss_nll, self.loss_confid = None, None
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input[0], dim=1)
        confidence = torch.sigmoid(input[1])

        # Make sure we don't have any numerical instability
        eps = 1e-12
        probs = torch.clamp(probs, 0.0 + eps, 1.0 - eps)
        confidence = torch.clamp(confidence, 0.0 + eps, 1.0 - eps)

        if self.half_random:
            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(self.device)
            conf = confidence * b + (1 - b)
        else:
            conf = confidence

        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(self.device)
        # Segmentation special case
        if self.task == "segmentation":
            labels_hot = labels_hot.permute(0, 3, 1, 2)
        probs_interpol = torch.log(conf * probs + (1 - conf) * labels_hot)
        self.loss_nll = nn.NLLLoss()(probs_interpol, target)
        self.loss_confid = torch.mean(-(torch.log(confidence)))
        total_loss = self.loss_nll + self.lbda * self.loss_confid

        # Update lbda
        if self.lbda_control:
            if self.loss_confid >= self.beta:
                self.lbda /= 0.99
            else:
                self.lbda /= 1.01
        return total_loss


# PYTORCH LOSSES LISTS
PYTORCH_LOSS = {"cross_entropy": nn.CrossEntropyLoss}

# CUSTOM LOSSES LISTS
CUSTOM_LOSS = {
    "selfconfid_mse": SelfConfidMSELoss,
    "selfconfid_tcpr": SelfConfidTCPRLoss,
    "selfconfid_bce": SelfConfidBCELoss,
    "focal": FocalLoss,
    "ranking": StructuredMAPRankingLoss,
    "ood_confidence": OODConfidenceLoss,
}
