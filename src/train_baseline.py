import argparse
import json
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from config import load_configs
from data import build_loaders
from models import build_model
from seed import set_seed

def next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator

def train_one_epoch(model, loader, loss_fn, optimizer, steps_per_epoch=None):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    if steps_per_epoch is None:
        batches = loader
    else:
        iterator = iter(loader)
        batches = range(steps_per_epoch)

    for batch in batches:
        if steps_per_epoch is None:
            x, y = batch
        else:
            (x, y), iterator = next_batch(iterator, loader)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(1) == y).sum().item()
        total_seen += batch_size

    return total_loss/total_seen, total_correct/total_seen

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
    steps_per_epoch = cfg["training"].get("steps_per_epoch")

    train_loader, val_loader, test_loader = build_loaders(cfg)
    model = build_model(cfg)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    
    start = time.time()
    best_val_acc = 0.0
    best_state = None
    history = []
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, steps_per_epoch)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        print(f"epoch {epoch:02d}/{cfg['training']['epochs']} | "
              f"train loss {train_loss:.4f} | "
              f"train acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} | "
              f"val acc {val_acc:.4f}")
        
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, loss_fn)

    results = {
        "method": "supervised",
        "seed": cfg["seed"],
        "num_labeled": cfg["data"]["num_labeled"],
        "model": cfg["model"]["name"],
        "steps_per_epoch": cfg["training"].get("steps_per_epoch"),
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "time(s)": (time.time() - start),
        "history": history
    }

    out_dir = Path(cfg["output"]["dir"]) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = cfg["training"].get("steps_per_epoch", "full")
    out_path = out_dir/f"supervised_{cfg['data']['num_labeled']}_steps{steps}_seed{cfg['seed']}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"test acc {test_acc:.4f}")
    print(f"saved to {out_path}")

if __name__ == "__main__":
    main()
