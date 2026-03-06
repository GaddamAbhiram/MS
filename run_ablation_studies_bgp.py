"""
ASFN Ablation Studies — Novel Component Analysis (BGP Dataset)
Runs each ablation for 50 epochs on BGP dataset and reports final test results.
Includes --class-weighted for BGP's imbalanced classes and --signed-smooth for heterophilic graph.
"""
import subprocess
import sys
import re
import json
from datetime import datetime

PYTHON = sys.executable
TRAIN_SCRIPT = "training/train.py"

BASE_ARGS = [
    "--model", "asfn",
    "--prefix", "./data/BGP_data/as",
    "--gpu", "-1",
    "--epochs", "50",
    "--patience", "50",
    "--class-weighted",
]

ABLATIONS = [
    ("Full ASFN + Signed Smooth (Baseline)",
     ["--smooth-lambda", "0.05", "--signed-smooth"]),

    ("w/o Signed Smoothness",
     ["--smooth-lambda", "0.05"]),

    ("w/o Adaptive Smoothness",
     ["--smooth-lambda", "0.0", "--signed-smooth"]),

    ("w/o Topology Encodings",
     ["--smooth-lambda", "0.05", "--signed-smooth", "--no-topo"]),

    ("w/o Smooth Decay (decay=0)",
     ["--smooth-lambda", "0.05", "--signed-smooth", "--smooth-decay", "0.0"]),

    ("w/o Global-Local Fusion",
     ["--smooth-lambda", "0.05", "--signed-smooth", "--no-global-fusion"]),

    ("w/ Smooth Decay 0.75",
     ["--smooth-lambda", "0.05", "--signed-smooth", "--smooth-decay", "0.75"]),
]


def run_ablation(name, extra_args):
    cmd = [PYTHON, TRAIN_SCRIPT] + BASE_ARGS + extra_args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        output = result.stdout + result.stderr
        match = re.search(r"The test Loss:\s*([\d.]+),\s*Acc:\s*([\d.]+)", output)
        if match:
            return float(match.group(1)), float(match.group(2))
        print(f"  [WARN] Could not parse results. Last 3 lines:")
        for line in output.strip().split("\n")[-3:]:
            print(f"    {line}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Timed out")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    print("=" * 65)
    print("  ASFN ABLATION STUDIES")
    print("  Dataset: BGP  |  Epochs: 50  |  Device: CPU")
    print("=" * 65)

    results = []
    for i, (name, extra_args) in enumerate(ABLATIONS, 1):
        print(f"\n[{i}/{len(ABLATIONS)}] {name} ...")
        outcome = run_ablation(name, extra_args)
        if outcome:
            test_loss, test_acc = outcome
            results.append({"ablation": name, "test_loss": test_loss, "test_acc": test_acc})
            print(f"  => Loss: {test_loss:.4f}  |  Acc: {test_acc:.4f}")
        else:
            results.append({"ablation": name, "test_loss": None, "test_acc": None})
            print(f"  => FAILED")

    # Summary table
    print("\n" + "=" * 65)
    print(f"  {'Ablation':<35} {'Loss':>10} {'Acc':>10} {'Delta':>10}")
    print("-" * 65)
    baseline_acc = results[0]["test_acc"] if results[0]["test_acc"] else None
    for r in results:
        loss = f"{r['test_loss']:.4f}" if r["test_loss"] is not None else "  FAILED"
        acc = f"{r['test_acc']:.4f}" if r["test_acc"] is not None else "  FAILED"
        if r["test_acc"] is not None and baseline_acc is not None:
            diff = r["test_acc"] - baseline_acc
            delta = "(base)" if diff == 0 else f"{diff:+.4f}"
        else:
            delta = ""
        print(f"  {r['ablation']:<35} {loss:>10} {acc:>10} {delta:>10}")
    print("=" * 65)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"ablation_results_bgp_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
