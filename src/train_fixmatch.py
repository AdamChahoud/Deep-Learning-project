import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from config import load_configs
from data import build_ssl_loaders
from models import build_model
from seed import set_seed

def next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator

def train_one_epoch(model, labeled_loader, unlabeled_loader, loss_fn, optimizer, threshold, lambda_u, steps_per_epoch):
    model.train()

    total_loss = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    total_correct = 0
    total_labeled = 0
    total_used = 0
    total_unlabeled = 0

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for _ in range(steps_per_epoch):
        (x_l, y_l), labeled_iter = next_batch(labeled_iter, labeled_loader)
        (u_w, u_s), unlabeled_iter = next_batch(unlabeled_iter, unlabeled_loader)
        
        logits_l = model(x_l)
        sup_loss = loss_fn(logits_l, y_l).mean()

        with torch.no_grad():
            logits_w = model(u_w)
            probs_w = torch.softmax(logits_w, dim=1)
            confidence, pseudo_label = probs_w.max(dim=1)
            mask = confidence.ge(threshold).float()

        logits_s = model(u_s)
        unsup_loss_all = loss_fn(logits_s, pseudo_label)
        unsup_loss = (unsup_loss_all * mask).mean()

        loss = sup_loss + lambda_u * unsup_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        labeled_batch = y_l.size(0)
        unlabeled_batch = u_w.size(0)

        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        total_correct += (logits_l.argmax(1) == y_l).sum().item()
        total_labeled += labeled_batch
        total_used += mask.sum().item()
        total_unlabeled += unlabeled_batch

    return {
        "loss": total_loss / steps_per_epoch,
        "sup_loss": total_sup_loss / steps_per_epoch,
        "unsup_loss": total_unsup_loss / steps_per_epoch,
        "train_acc": total_correct / total_labeled,
        "used_unlabeled_frac": total_used / total_unlabeled,
    }



def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(1) == y).sum().item()
            total_seen += batch_size

    return total_loss/total_seen, total_correct/total_seen



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_configs(args.config)
    set_seed(cfg["seed"])

    labeled_loader, unlabeled_loader, val_loader, test_loader = build_ssl_loaders(cfg, method="fixmatch")
    model = build_model(cfg)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    eval_loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    threshold = cfg["fixmatch"]["threshold"]
    lambda_u = cfg["fixmatch"]["lambda_u"]
    steps_per_epoch = cfg["training"]["steps_per_epoch"]

    best_val_acc = 0.0
    best_state = None
    history = []

    start = time.time()
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics = train_one_epoch(model, labeled_loader, unlabeled_loader, loss_fn, optimizer, threshold, lambda_u, steps_per_epoch)
        val_loss, val_acc = evaluate(model, val_loader, eval_loss_fn)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "sup_loss": train_metrics["sup_loss"],
            "unsup_loss": train_metrics["unsup_loss"],
            "train_acc": train_metrics["train_acc"],
            "used_unlabeled_frac": train_metrics["used_unlabeled_frac"],
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(f"epoch {epoch:02d}/{cfg['training']['epochs']} | "
              f"loss {train_metrics['loss']:.4f} | "
              f"sup {train_metrics['sup_loss']:.4f} | "
              f"unsup {train_metrics['unsup_loss']:.4f} | "
              f"used_u {train_metrics['used_unlabeled_frac']:.3f} | "
              f"train acc {train_metrics['train_acc']:.4f} | "
              f"val acc {val_acc:.4f}")
        
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, eval_loss_fn)

    results = {
        "method": "fixmatch",
        "seed": cfg["seed"],
        "num_labeled": cfg["data"]["num_labeled"],
        "model": cfg["model"]["name"],
        "threshold": threshold,
        "lambda_u": lambda_u,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "time(s)": time.time() - start,
        "history": history,
    }
        
    out_dir = Path(cfg["output"]["dir"]) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"fixmatch_{cfg['data']['num_labeled']}"
                          f"_t{str(threshold).replace('.', '')}"
                          f"_seed{cfg['seed']}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Test acc: {test_acc:.4f}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()