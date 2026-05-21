import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

def load_result(path):
    with open(path, "r") as f:
        return json.load(f)
    
def get(history, key):
    return [row[key] for row in history]

def steps(result):
    return result.get("steps_per_epoch") or "full"

def tag(result):
    name = f"{result['method']}_{result['num_labeled']}_steps{steps(result)}"

    if result["method"] == "fixmatch":
        threshold = str(result["threshold"]).replace(".", "")
        name += f"_t{threshold}"

    return name

def key(result):
    return (
        result["method"],
        result["num_labeled"],
        steps(result),
        result.get("threshold")
    )
    
def index_results(results):
    return {key(r): r for r in results}


def plot_accs(result, out_dir):
    history = result["history"]
    epochs = get(history, "epoch")

    plt.figure()
    plt.plot(epochs, get(history, "train_acc"), label="train acc")
    plt.plot(epochs, get(history, "val_acc"), label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(tag(result))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag(result)}_acc.png")
    plt.close()


def plot_losses(result, out_dir):
    history = result["history"]
    epochs = get(history, "epoch")

    plt.figure()
    plt.plot(epochs, get(history, "train_loss"), label="train loss")
    plt.plot(epochs, get(history, "val_loss"), label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(tag(result))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag(result)}_loss.png")
    plt.close()


def plot_fixmatch_usage(result, out_dir):
    if result["method"] != "fixmatch":
        return
    
    history = result["history"]
    epochs = get(history, "epoch")

    plt.figure()
    plt.plot(epochs, get(history, "used_unlabeled_frac"))
    plt.xlabel("epoch")
    plt.ylabel("unlabeled used fraction")
    plt.title(tag(result))
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag(result)}_used_unlabeled.png")
    plt.close()


def plot_acc_vs_labels(results_by_key, out_dir, step_count, threshold=0.8):
    labels = [250, 1000, 5000]
    sup_acc = []
    fix_acc = []
    used_labels = []

    for n in labels:
        sup = results_by_key.get(("supervised", n, step_count, None))
        fix = results_by_key.get(("fixmatch", n, step_count, threshold))

        if sup is None or fix is None:
            continue

        used_labels.append(n)
        sup_acc.append(sup["test_acc"])
        fix_acc.append(fix["test_acc"])

    if not used_labels:
        return
    
    plt.figure()
    plt.plot(used_labels, sup_acc, marker="o", label="Supervised")
    plt.plot(used_labels, fix_acc, marker="o", label="FixMatch")
    plt.xlabel("number of labeled samples")
    plt.ylabel("test accuracy")
    plt.title(f"Supervised vs FixMatch, {step_count} steps/epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_vs_labels_steps{step_count}.png")
    plt.close()

def plot_acc_vs_steps(results_by_key, out_dir, num_labeled=1000, threshold=0.8):
    step_counts = [50, 100, 400]
    sup_acc = []
    fix_acc = []
    used_steps = []

    for s in step_counts:
        sup = results_by_key.get(("supervised", num_labeled, s, None))
        fix = results_by_key.get(("fixmatch", num_labeled, s, threshold))

        if sup is None or fix is None:
            continue
        used_steps.append(s)
        sup_acc.append(sup["test_acc"])
        fix_acc.append(fix["test_acc"])

    if not used_steps:
        return
    
    plt.figure()
    plt.plot(used_steps, sup_acc, marker="o", label="Supervised")
    plt.plot(used_steps, fix_acc, marker="o", label="FixMatch")
    plt.xlabel("steps per epoch")
    plt.ylabel("test accuracy")
    plt.title(f"Supervised vs FixMatch, {num_labeled} labels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_vs_steps_labels{num_labeled}.png")
    plt.close()


def plot_threshold_ablation(results_by_key, out_dir, num_labeled=1000, step_count=100):
    thresholds = [0.7, 0.8, 0.9, 0.95]
    accs = []
    usage = []
    used_thresholds = []

    for t in thresholds:
        result = results_by_key.get(("fixmatch", num_labeled, step_count, t)) 

        if result is None:
            continue

        used_thresholds.append(t)
        accs.append(result["test_acc"])
        usage.append(result["history"][-1]["used_unlabeled_frac"])

    if not used_thresholds:
        return
    
    plt.figure()
    plt.plot(used_thresholds, accs, marker="o")
    plt.xlabel("confidence threshold")
    plt.ylabel("test accuracy")
    plt.title(f"FixMatch threshold accuracy, {num_labeled} labels")
    plt.tight_layout()
    plt.savefig(out_dir / f"threshold_vs_acc_{num_labeled}_steps{step_count}.png")
    plt.close()


    plt.figure()
    plt.plot(used_thresholds, usage, marker="o")
    plt.xlabel("confidence threshold")
    plt.ylabel("final unlabeled usage")
    plt.title(f"FixMatch threshold usage, {num_labeled} labels")
    plt.tight_layout()
    plt.savefig(out_dir / f"threshold_vs_usage_{num_labeled}_steps{step_count}.png")
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--out", default="outputs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [load_result(path) for path in args.json_files]
    results_by_key = index_results(results)

    for result in results:
        plot_accs(result, out_dir)
        plot_losses(result, out_dir)
        plot_fixmatch_usage(result, out_dir)


    plot_acc_vs_labels(results_by_key, out_dir, step_count=50)
    plot_acc_vs_labels(results_by_key, out_dir, step_count=100)
    plot_acc_vs_steps(results_by_key, out_dir, num_labeled=1000)
    plot_threshold_ablation(results_by_key, out_dir, num_labeled=1000, step_count=100)

if __name__ == "__main__":
    main()