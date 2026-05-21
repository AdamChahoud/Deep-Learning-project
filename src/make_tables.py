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

def print_threshold_ablation(results):
    labels = 1000
    steps = 100
    thresholds = [0.7, 0.8, 0.9, 0.95]

    # FIXMATCH
    print("\nFixMatch threshold ablation")
    print("| Labels | Steps | Threshold | Test acc | Final unlabeled usage |")
    print("|-------:|------:|----------:|---------:|----------------------:|")

    for threshold in thresholds:
        r = find_result(results, "fixmatch", labels, steps, threshold=threshold)
        if r is None:
            continue

        final_used = r["history"][-1]["used_unlabeled_frac"]
        print(f"| {labels:<6} | {steps:<5} | {threshold:<9.2f} | {r['test_acc']:<8.4f} | {final_used:<21.4f} |")

    # MIXMATCH
    print("\n\nMixMatch threshold ablation")
    print("| Labels | Steps | Threshold | Test acc | Final unlabeled usage |")
    print("|-------:|------:|----------:|---------:|----------------------:|")

    for threshold in thresholds:
        r = find_result(results, "mixmatch", labels, steps, threshold=threshold)
        if r is None:
            continue

        final_used = r["history"][-1]["used_unlabeled_frac"]
        print(f"| {labels:<6} | {steps:<5} | {threshold:<9.2f} | {r['test_acc']:<8.4f} | {final_used:<21.4f} |")

def print_comparison(results):
    labels_list = [250, 1000, 5000]
    steps_list = ["full", 50, 100, 400]

    print("\nSupervised | FixMatch | MixMatch")
    print("| Labels | Steps/epoch | Supervised acc | FixMatch acc | FixMatch - Supervised | MixMatch acc | MixMatch - Supervised |")
    print("|-------:|------------:|---------------:|-------------:|----------------------:|-------------:|----------------------:|")

    for labels in labels_list:
        for steps in steps_list:
            sup = find_result(results, "supervised",labels,steps)
            fix = find_result(results, "fixmatch",labels,steps, threshold=0.8)
            mix = find_result(results, "mixmatch",labels,steps, threshold=0.8)

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

            print(f"| {labels:<6} | {str(steps):<11} | {sup_text:<14} | {fix_text:<12} | {Fdiff_text:<21} | {mix_text:<12} | {Mdiff_text:<21} |")
def main():
    results = load_all_results()
    print_comparison(results)
    print_threshold_ablation(results)


if __name__ == "__main__":
    main()