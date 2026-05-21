# -*- coding: utf-8 -*-
"""
MixMatch: A Holistic Approach to Semi-Supervised Learning
Berthelot et al., NeurIPS 2019  --  https://arxiv.org/abs/1905.02249

Algorithm (Algorithm 1 from paper):
  Given a labeled batch X and unlabeled batch U:
    1. Augment each labeled example once:          x_b  = Augment(x_b)
    2. Augment each unlabeled example K times:     u_bk = Augment(u_b)
    3. Average model predictions over K views:     q_b  = (1/K) sum p(y|u_bk)
    4. Sharpen with temperature T:                 q_b  = Sharpen(q_b, T)
    5. Build Xhat = {(x_b, p_b)}, Uhat = {(u_bk, q_b)}
    6. W = ShuffleConcat(Xhat, Uhat)
    7. X' = MixUp(Xhat_i, W_i)
    8. U' = MixUp(Uhat_i, W_{i+|Xhat|})
    9. L_X = (1/|X'|)  sum H(p, p_model(y|x))      cross-entropy
       L_U = (1/|U'|)  sum ||q - p_model(y|u)||^2  MSE
       L   = L_X + lambda_U(t) * L_U

Hyperparameters (CIFAR-10, Section 4):
  T=0.5, K=2, alpha=0.75, lambda_U=75, weight_decay=4e-4
  Adam lr=2e-3, lambda_U linearly ramped up over 16 000 steps

Usage:
  python mixmatch.py --config ../configs/mixmatch_1000.yaml
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import load_configs
from data   import build_ssl_loaders
from models import build_model
from seed   import set_seed


# ============================================================
# MixMatch core functions
# ============================================================

def sharpen(p: torch.Tensor, T: float) -> torch.Tensor:
    """
    Temperature sharpening (Eq. 7 in paper).
    Sharpen(p, T)_i = p_i^(1/T) / sum_j p_j^(1/T)
    As T -> 0 the distribution approaches one-hot.
    """
    p_t = p.pow(1.0 / T)
    return p_t / p_t.sum(dim=1, keepdim=True)


def mixup(x1: torch.Tensor, p1: torch.Tensor,
          x2: torch.Tensor, p2: torch.Tensor,
          alpha: float) -> tuple:
    """
    MixUp with the lambda' = max(lambda, 1-lambda) modification from MixMatch (Eq. 8-11).
    Ensures x' is always closer to x1 than x2, keeping labeled/unlabeled batches separate.
    """
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    x_mix = lam * x1 + (1.0 - lam) * x2
    p_mix = lam * p1 + (1.0 - lam) * p2
    return x_mix, p_mix


def mixmatch_batch(model: nn.Module,
                   labeled_imgs:    torch.Tensor,
                   labels_onehot:   torch.Tensor,
                   unlabeled_views: list,
                   T: float,
                   K: int,
                   alpha: float,
                   device: torch.device):
    """
    Full MixMatch preprocessing on one mini-batch (Algorithm 1).

    Args:
        model           : classifier in train mode
        labeled_imgs    : (B, C, H, W) augmented labeled images
        labels_onehot   : (B, num_classes) one-hot label tensor
        unlabeled_views : list of K tensors, each (B, C, H, W)
        T, K, alpha     : MixMatch hyperparameters
        device          : torch device

    Returns:
        mixed_labeled_x  : (B, C, H, W)
        mixed_labeled_p  : (B, num_classes)
        mixed_unlabeled_x: (K*B, C, H, W)
        mixed_unlabeled_p: (K*B, num_classes)
    """
    B = labeled_imgs.size(0)

    # Steps 3-4: label guessing for unlabeled data
    with torch.no_grad():
        all_views = torch.cat(unlabeled_views, dim=0)           # (K*B, C, H, W)
        all_preds = torch.softmax(model(all_views), dim=1)      # (K*B, num_classes)
        all_preds = all_preds.view(K, B, -1).mean(dim=0)       # (B, num_classes)
        guessed_labels = sharpen(all_preds, T)                  # (B, num_classes)

    # Step 5: build Xhat and Uhat
    xhat_x = labeled_imgs
    xhat_p = labels_onehot
    uhat_x = torch.cat(unlabeled_views, dim=0)                  # (K*B, C, H, W)
    uhat_p = guessed_labels.repeat(K, 1)                        # (K*B, num_classes)

    # Step 6: W = ShuffleConcat(Xhat, Uhat)
    all_x = torch.cat([xhat_x, uhat_x], dim=0)
    all_p = torch.cat([xhat_p, uhat_p], dim=0)
    idx   = torch.randperm(all_x.size(0), device=device)
    W_x   = all_x[idx]
    W_p   = all_p[idx]

    # Step 7: X' = MixUp(Xhat_i, W_i)
    mixed_labeled_x, mixed_labeled_p = [], []
    for i in range(B):
        mx, mp = mixup(xhat_x[i], xhat_p[i], W_x[i], W_p[i], alpha)
        mixed_labeled_x.append(mx)
        mixed_labeled_p.append(mp)
    mixed_labeled_x = torch.stack(mixed_labeled_x)
    mixed_labeled_p = torch.stack(mixed_labeled_p)

    # Step 8: U' = MixUp(Uhat_i, W_{i+B})
    mixed_unlabeled_x, mixed_unlabeled_p = [], []
    KB = uhat_x.size(0)
    for i in range(KB):
        mx, mp = mixup(uhat_x[i], uhat_p[i], W_x[B + i], W_p[B + i], alpha)
        mixed_unlabeled_x.append(mx)
        mixed_unlabeled_p.append(mp)
    mixed_unlabeled_x = torch.stack(mixed_unlabeled_x)
    mixed_unlabeled_p = torch.stack(mixed_unlabeled_p)

    return mixed_labeled_x, mixed_labeled_p, mixed_unlabeled_x, mixed_unlabeled_p


# ============================================================
# Training helpers
# ============================================================

def next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def linear_rampup(current_step: int, rampup_length: int) -> float:
    """Linearly ramp lambda_U from 0 to 1 over rampup_length steps."""
    if rampup_length == 0:
        return 1.0
    return min(1.0, current_step / rampup_length)


def train_one_epoch(model, labeled_loader, unlabeled_loader, optimizer,
                    T, K, alpha, lambda_u, rampup_length,
                    steps_per_epoch, num_classes, device, global_step):
    model.train()

    total_loss = 0.0
    total_lx   = 0.0
    total_lu   = 0.0
    total_correct = 0
    total_labeled = 0

    labeled_iter   = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    eps = 1e-8

    for _ in range(steps_per_epoch):
        (x_l, y_l), labeled_iter     = next_batch(labeled_iter,   labeled_loader)
        (u_a, u_b), unlabeled_iter   = next_batch(unlabeled_iter, unlabeled_loader)

        x_l, y_l = x_l.to(device), y_l.to(device)
        u_a, u_b = u_a.to(device),  u_b.to(device)

        B             = x_l.size(0)
        labels_onehot = F.one_hot(y_l, num_classes).float()
        unlabeled_views = [u_a, u_b]   # K=2 views from PairedUnlabeledDataset

        # Linear ramp-up of lambda_U (Section 3.5)
        lambda_u_eff = lambda_u * linear_rampup(global_step, rampup_length)

        # MixMatch preprocessing (Algorithm 1)
        mixed_lx, mixed_lp, mixed_ux, mixed_up = mixmatch_batch(
            model, x_l, labels_onehot, unlabeled_views, T, K, alpha, device
        )

        # Forward passes
        logits_x = model(mixed_lx)
        probs_x  = torch.softmax(logits_x, dim=1)
        logits_u = model(mixed_ux)
        probs_u  = torch.softmax(logits_u, dim=1)

        # L_X: soft cross-entropy on mixed labeled batch
        L_x = -(mixed_lp * torch.log(probs_x + eps)).sum(dim=1).mean()

        # L_U: MSE (Brier score) on mixed unlabeled batch
        L_u = F.mse_loss(probs_u, mixed_up)

        loss = L_x + lambda_u_eff * L_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        total_lx      += L_x.item()
        total_lu      += L_u.item()
        total_correct += (logits_x.argmax(1) == y_l).sum().item()
        total_labeled += B
        global_step   += 1

    return {
        "loss":      total_loss / steps_per_epoch,
        "lx":        total_lx   / steps_per_epoch,
        "lu":        total_lu   / steps_per_epoch,
        "train_acc": total_correct / total_labeled,
    }, global_step


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_seen    = 0

    with torch.no_grad():
        for x, y in loader:
            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            loss    = loss_fn(logits, y)
            batch_size = y.size(0)
            total_loss    += loss.item() * batch_size
            total_correct += (logits.argmax(1) == y).sum().item()
            total_seen    += batch_size

    return total_loss / total_seen, total_correct / total_seen


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MixMatch semi-supervised training on CIFAR-10"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_configs(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MixMatch] device={device}  num_labeled={cfg['data']['num_labeled']}")

    labeled_loader, unlabeled_loader, val_loader, test_loader = build_ssl_loaders(
        cfg, method="mixmatch"
    )
    model = build_model(cfg).to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    # Read MixMatch hyperparameters from config.
    # The YAML uses a "mixmatch" section; falls back to "fixmatch" if absent.
    mm_cfg        = cfg.get("mixmatch", cfg.get("fixmatch", {}))
    T             = mm_cfg.get("temperature", 0.5)
    K             = mm_cfg.get("K", 2)
    alpha         = mm_cfg.get("alpha", 0.75)
    lambda_u      = mm_cfg.get("lambda_u", 75.0)
    rampup_length = mm_cfg.get("rampup_length", 16000)

    epochs          = cfg["training"]["epochs"]
    steps_per_epoch = cfg["training"].get("steps_per_epoch", len(labeled_loader))
    num_classes     = cfg["data"]["num_classes"]

    best_val_acc = 0.0
    best_state   = None
    history      = []
    global_step  = 0

    start = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model, labeled_loader, unlabeled_loader, optimizer,
            T, K, alpha, lambda_u, rampup_length,
            steps_per_epoch, num_classes, device, global_step,
        )

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())

        history.append({
            "epoch":     epoch,
            "train_loss": train_metrics["loss"],
            "lx":        train_metrics["lx"],
            "lu":        train_metrics["lu"],
            "train_acc": train_metrics["train_acc"],
            "val_loss":  val_loss,
            "val_acc":   val_acc,
        })

        print(f"epoch {epoch:02d}/{epochs} | "
              f"loss {train_metrics['loss']:.4f} | "
              f"Lx {train_metrics['lx']:.4f} | "
              f"Lu {train_metrics['lu']:.4f} | "
              f"train acc {train_metrics['train_acc']:.4f} | "
              f"val acc {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

    results = {
        "method":      "mixmatch",
        "seed":        cfg["seed"],
        "num_labeled": cfg["data"]["num_labeled"],
        "model":       cfg["model"]["name"],
        "temperature": T,
        "alpha":       alpha,
        "lambda_u":    lambda_u,
        "best_val_acc": best_val_acc,
        "test_loss":   test_loss,
        "test_acc":    test_acc,
        "time(s)":     time.time() - start,
        "history":     history,
    }

    out_dir = Path(cfg["output"]["dir"]) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mixmatch_{cfg['data']['num_labeled']}_seed{cfg['seed']}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Test acc: {test_acc:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
