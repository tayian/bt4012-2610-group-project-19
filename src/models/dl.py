"""W1-C3 shared PyTorch training loop for M4 (MLP) and M7 (1D-CNN).

* weighted BCE (pos_weight = negatives / positives) or focal loss (gamma 2)
* early stopping on validation PR-AUC (10% stratified inner split of the
  training fold; the held-out CV fold is never seen during training)
* returns held-out fold scores for the OOF file (4.4 contract)
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

SEED = 4012
torch.set_num_threads(2)


def set_seed(s=SEED):
    torch.manual_seed(s)
    np.random.seed(s)


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017) with alpha balancing."""

    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = y * p + (1 - y) * (1 - p)
        a = y * self.alpha + (1 - y) * (1 - self.alpha)
        return (a * (1 - pt) ** self.gamma * bce).mean()


def make_loss(name, y):
    if name == "focal":
        return FocalLoss()
    pw = torch.tensor((y == 0).sum() / max(1, (y == 1).sum()), dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def _batches(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(model, inputs, bs=4096):
    model.eval()
    n = len(inputs[0])
    out = []
    for b in _batches(n, bs, False, None):
        out.append(torch.sigmoid(model(*[torch.as_tensor(x[b]) for x in inputs])).numpy())
    return np.concatenate(out)


def train_model(make_model, inputs, y, loss="bce", lr=1e-3, bs=512, max_epochs=30,
                patience=4, seed=SEED, verbose=True, augment=None):
    """inputs: list of np arrays (same first dim). augment(batch_inputs, y_batch, model)
    may return extra (inputs, y) to append to each batch (used for adversarial training)."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = make_loss(loss, y[tr])
    best, best_state, bad = -1, None, 0
    tr_in = [x[tr] for x in inputs]
    ytr = y[tr]
    for ep in range(max_epochs):
        t = time.time()
        model.train()
        for b in _batches(len(tr), bs, True, rng):
            xb = [torch.as_tensor(x[b]) for x in tr_in]
            yb = torch.as_tensor(ytr[b])
            if augment is not None:
                ex, ey = augment(xb, yb, model)
                if ex is not None:
                    model.train()
                    xb = [torch.cat([a, e]) for a, e in zip(xb, ex)]
                    yb = torch.cat([yb, ey])
            opt.zero_grad()
            l = crit(model(*xb), yb)
            l.backward()
            opt.step()
        ap = average_precision_score(y[va], predict(model, [x[va] for x in inputs]))
        if verbose:
            print(f"    ep {ep:2d} val PR-AUC {ap:.4f}  ({time.time() - t:.0f}s)", flush=True)
        if ap > best + 1e-4:
            best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best
