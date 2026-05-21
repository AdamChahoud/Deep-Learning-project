import json
from pathlib import Path

RESULT_DIR = Path("outputs/results")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def load_all_results():
    return [load_json(p) for p in sorted(RESULT_DIR.glob("*.json"))]

def find_result(results, method, labels, steps, threshold=None):
    matches = []

    for r in results:
        if r["method"] != method:
            continue
        if r["num_labeled"] != labels:
            continue
        if (r.get("steps_per_epoch") or "full") != steps:
            continue
        if threshold is not None and r.get("threshold") != threshold:
            continue
        if threshold is None and "threshold" in r:
            continue
        
        matches.append(r)

    if len(matches) == 0:
        return None
    
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {method}, labels={labels}, steps={steps}, threshold={threshold}")
    
    return matches[0]

def print_supervised_fixmatch(results):
    labels_list = [250, 1000, 5000]
    steps_list = ["full", 50, 100, 400]

    print("\nSupervised vs FixMatch")
    print("| Labels | Steps/epoch | Supervised acc | FixMatch acc | Difference |")
    print("|---:|---:|---:|---:|---:|")

    for labels in labels_list:
        for steps in steps_list:
            sup = find_result(results, "supervised", labels, steps)
            fix = find_result(results, "fixmatch", labels, steps, threshold=0.8)

            if sup is None and fix is None:
                continue

            sup_acc = sup["test_acc"] if sup is not None else None
            fix_acc = fix["test_acc"] if fix is not None else None

            if sup_acc is None:
                sup_text = "-"
                diff_text = "-"
            else:
                sup_text = f"{sup_acc:.4f}"

            if fix_acc is None:
                fix_text = "-"
                diff_text = "-"
            else:
                fix_text = f"{fix_acc:.4f}"

            if sup_acc is not None and fix_acc is not None:
                diff_text = f"{fix_acc - sup_acc:+.4f}"

            print(f"| {labels} | {steps} | {sup_text} | {fix_text} | {diff_text} |")

def print_threshold_ablation(results):
    labels = 1000
    steps = 100
    thresholds = [0.7, 0.8, 0.9, 0.95]

    print("\nFixMatch threshold ablation")
    print("| Labels | Steps | Threshold | Test acc | Final unlabeled usage |")
    print("|-----:|----:|-----:|-------:|-------:|")

    for threshold in thresholds:
        r = find_result(results, "fixmatch", labels, steps, threshold=threshold)
        if r is None:
            continue

        final_used = r["history"][-1]["used_unlabeled_frac"]

        print(f"| {labels} | {steps} | {threshold:.2f} | {r['test_acc']:.4f} | {final_used:.4f} |")


def main():
    results = load_all_results()
    print_supervised_fixmatch(results)
    print_threshold_ablation(results)


if __name__ == "__main__":
    main()