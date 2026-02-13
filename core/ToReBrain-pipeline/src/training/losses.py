import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMBCEWithLogitsLoss(nn.Module):
    """Online Hard Example Mining BCE for extreme class imbalance.

    Keeps all positive voxels, and only the hardest negatives (top-k by BCE loss).
    """

    def __init__(
        self,
        neg_fraction: float = 0.1,
        min_neg: int = 1024,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
    ):
        super().__init__()
        self.neg_fraction = float(neg_fraction)
        self.min_neg = int(min_neg)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)

    def forward(self, logits, targets):
        # logits/targets: (B, 1, D, H, W)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = loss.flatten(1)
        t = targets.flatten(1)

        pos_mask = t > 0.5
        neg_mask = ~pos_mask

        pos_loss = loss[pos_mask]
        neg_loss = loss[neg_mask]

        # keep all positives (if any)
        pos_mean = pos_loss.mean() if pos_loss.numel() > 0 else loss.new_tensor(0.0)

        # keep hardest negatives
        if neg_loss.numel() > 0:
            frac = float(max(0.0, min(1.0, self.neg_fraction)))
            k = int(max(self.min_neg, round(frac * float(neg_loss.numel()))))
            k = int(min(k, neg_loss.numel()))
            hard_neg = neg_loss.topk(k, largest=True).values
            neg_mean = hard_neg.mean()
        else:
            neg_mean = loss.new_tensor(0.0)

        return self.pos_weight * pos_mean + self.neg_weight * neg_mean


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0, pos_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        # Use functional BCE so pos_weight can be placed on the same device as logits
        # (BCEWithLogitsLoss stores pos_weight as a Tensor attribute that may remain on CPU).
        self.pos_weight = float(pos_weight)
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        pos_w = logits.new_tensor([self.pos_weight])
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)
        probs = torch.sigmoid(logits)
        probs_flat = torch.flatten(probs, start_dim=1)
        targets_flat = torch.flatten(targets, start_dim=1)
        inter = (probs_flat * targets_flat).sum(1)
        den = probs_flat.sum(1) + targets_flat.sum(1)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice.mean()
        return (self.bce_weight * bce) + dice_loss


class DiceOHEMBCELoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
        neg_fraction: float = 0.1,
        min_neg: int = 1024,
        bce_weight: float = 1.0,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
    ):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = float(bce_weight)
        self.ohem = OHEMBCEWithLogitsLoss(
            neg_fraction=neg_fraction,
            min_neg=min_neg,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
        )

    def forward(self, logits, targets):
        ohem_bce = self.ohem(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = torch.flatten(probs, start_dim=1)
        targets_flat = torch.flatten(targets, start_dim=1)
        inter = (probs_flat * targets_flat).sum(1)
        den = probs_flat.sum(1) + targets_flat.sum(1)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss + self.bce_weight * ohem_bce


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = torch.flatten(probs, start_dim=1)
        targets_flat = torch.flatten(targets, start_dim=1)

        # Dice part
        inter = (probs_flat * targets_flat).sum(1)
        den = probs_flat.sum(1) + targets_flat.sum(1)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice.mean()

        # Focal BCE part
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (self.alpha * (1 - pt) ** self.gamma * ce).mean()

        return dice_loss + focal


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = torch.flatten(probs, start_dim=1)
        targets_flat = torch.flatten(targets, start_dim=1)

        tp = (probs_flat * targets_flat).sum(1)
        fp = (probs_flat * (1 - targets_flat)).sum(1)
        fn = ((1 - probs_flat) * targets_flat).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = torch.flatten(probs, start_dim=1)
        targets_flat = torch.flatten(targets, start_dim=1)

        tp = (probs_flat * targets_flat).sum(1)
        fp = (probs_flat * (1 - targets_flat)).sum(1)
        fn = ((1 - probs_flat) * targets_flat).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_t = (1 - tversky) ** self.gamma
        return focal_t.mean()


class TverskyOHEMBCELoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        neg_fraction: float = 0.1,
        min_neg: int = 1024,
        bce_weight: float = 1.0,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce_weight = float(bce_weight)
        self.ohem = OHEMBCEWithLogitsLoss(
            neg_fraction=neg_fraction,
            min_neg=min_neg,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
        )

    def forward(self, logits, targets):
        ohem_bce = self.ohem(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = torch.flatten(probs, start_dim=1)
        targets_flat = torch.flatten(targets, start_dim=1)

        tp = (probs_flat * targets_flat).sum(1)
        fp = (probs_flat * (1 - targets_flat)).sum(1)
        fn = ((1 - probs_flat) * targets_flat).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = 1 - tversky.mean()
        return tversky_loss + self.bce_weight * ohem_bce


class DiceCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.smooth = smooth

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        target_1h = torch.nn.functional.one_hot(target, num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        probs_flat = probs.flatten(2)
        target_flat = target_1h.flatten(2)
        inter = (probs_flat * target_flat).sum(-1)
        den = probs_flat.sum(-1) + target_flat.sum(-1)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        return ce + (1 - dice.mean())
