import json
from pathlib import Path

RESULT_DIR = Path("outputs/results")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def load_all_results():
    return [load_json(p) for p in sorted(RESULT_DIR.glob("*.json"))]

def find_result(results, method, labels, steps, threshold=None, augment=None):
    if method == "fixmatch" and augment is None:
        augment = "RandAugment"
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
        if augment is not None and r.get("augment_strategy") != augment:
            continue
        
        matches.append(r)

    if len(matches) == 0:
        return None
    
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {method}, labels={labels}, steps={steps}, threshold={threshold}")
    
    return matches[0]

def print_fixmatch_threshold_data(results):
    labels = 1000
    steps = 100
    thresholds = [0.7, 0.8, 0.9, 0.95]

    print("\nFixMatch threshold ablation")
    print("| Labels | Steps | Threshold | Test acc | Final unlabeled usage | Time (s) |")
    print("|--------|-------|-----------|----------|-----------------------|----------|")

    for threshold in thresholds:
        r = find_result(results, "fixmatch", labels, steps, threshold=threshold)
        if r is None:
            continue

        final_used = r["history"][-1]["used_unlabeled_frac"]
        run_time = r.get("time(s)", 0.0)
        print(f"| {labels:<6} | {steps:<5} | {threshold:<9.2f} | {r['test_acc']:<8.4f} | {final_used:<21.4f} | {run_time:<8.1f} |")

def print_mixmatch_label_data(results):
    labels_list = [250, 500, 1000, 5000]
    steps = "full"


    print(f"\nMixMatch Data Scale Analysis (Steps/epoch: {steps})")
    print("| Labels | Steps | Test acc | Time (s) |")
    print("|--------|-------|----------|----------|")

    for labels in labels_list:
        r = find_result(results, "mixmatch", labels, steps, threshold=None)
        if r is None:
            continue

        test_acc = r["test_acc"]
        run_time = r.get("time(s)", 0.0)

        # Print layout with spacing padding matching the column headers
        print(f"| {labels:<6} | {str(steps):<5} | {test_acc:<8.4f} | {run_time:<8.1f} |")
              
def print_comparison(results):
    labels_list = [250, 1000, 5000]
    steps_list = ["full", 50, 100, 400]

    print("\nSupervised | FixMatch | MixMatch")
    print("| Labels | Steps | Supervised acc | FixMatch acc | FixMatch - Supervised | MixMatch acc | MixMatch - Supervised |")
    print("|--------|-------|----------------|--------------|-----------------------|--------------|-----------------------|")

    for labels in labels_list:
        for steps in steps_list:
            sup = find_result(results, "supervised",labels,steps)
            fix = find_result(results, "fixmatch",labels,steps, threshold=0.8)
            mix = find_result(results, "mixmatch",labels,steps, threshold=None)

            if sup is None and fix is None and mix is None:
                continue

            sup_acc = sup["test_acc"] if sup is not None else None
            fix_acc = fix["test_acc"] if fix is not None else None
            mix_acc = mix["test_acc"] if mix is not None else None

            sup_text, Fdiff_text, Mdiff_text = "-", "-", "-"
            fix_text, mix_text = "-", "-"

            if sup_acc is not None:
                sup_text = f"{sup_acc:.4f}"

            if fix_acc is not None:
                fix_text = f"{fix_acc:.4f}"
                if sup_acc is not None:
                    Fdiff_text = f"{fix_acc - sup_acc:+.4f}"

            if mix_acc is not None:
                mix_text = f"{mix_acc:.4f}"
                if sup_acc is not None:
                    Mdiff_text = f"{mix_acc - sup_acc:+.4f}"

            print(f"| {labels:<6} | {str(steps):<5} | {sup_text:<14} | {fix_text:<12} | {Fdiff_text:<21} | {mix_text:<12} | {Mdiff_text:<21} |")

def print_augmentation_ablation_data(results):
    labels = 1000
    steps = 400
    threshold = 0.8
    
    # The strategies you want to compare
    augments = ["AutoAugment", "RandAugment", "Cutout", "Trivial", "AugMix"]

    print(f"\nFixMatch Augmentation Strategy Analysis (Labels: {labels} | Steps: {steps} | Threshold: {threshold})")
    print("| Augmentation Strategy | Test Acc | Time (s) | Final Unlabeled Usage |")
    print("|-----------------------|----------|----------|-----------------------|")

    for aug in augments:
        # Pass the augment parameter directly into our updated find_result function
        r = find_result(results, "fixmatch", labels, steps, threshold=threshold, augment=aug)
        if r is None:
            # If a specific strategy run isn't finished or found yet, display a placeholder row
            print(f"| {aug:<21} | {'-':<8} | {'-':<8} | {'-':<21} |")
            continue

        test_acc = r["test_acc"]
        run_time = r.get("time(s)", 0.0)
        
        # Safely fetch the final unlabeled utilization fraction if it exists
        final_used = r["history"][-1]["used_unlabeled_frac"] if "history" in r and r["history"] else 0.0

        print(f"| {aug:<21} | {test_acc:<8.4f} | {run_time:<8.1f} | {final_used:<21.4f} |")

def main():
    results = load_all_results()
    print_comparison(results)
    print_fixmatch_threshold_data(results)
    print_mixmatch_label_data(results)
    print_augmentation_ablation_data(results)


if __name__ == "__main__":
    main()