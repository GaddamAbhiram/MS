#!/usr/bin/env python3
"""
SoftGNN on BGP Dataset - 10 Random Seed Experiments
Runs the same optimal hyperparameters across 10 different random seeds
to establish accuracy ± std with statistical significance.
"""

import subprocess
import json
import os
import numpy as np
from datetime import datetime

# 10 random seeds for statistical evaluation
SEEDS = [42, 123, 256, 512, 1024, 2048, 3407, 4096, 7777, 9999]

# Optimal hyperparameters (matched with ASFN for fair comparison)
OPTIMAL_PARAMS = {
    "num_hidden": 32,
    "num_layers": 2,
    "lr": 0.01,
    "in_drop": 0.3,
    "softgnn_feat_drop": 0.3,
    "softgnn_attn_dim": 16,
    "batch_size": 512,
    "smooth_lambda": 0.05,
    "epochs": 50,
    "patience": 50,
}


def build_command(seed, gpu=0):
    """Build the training command for a given seed"""
    cmd = [
        "python", "training/train.py",
        "--model", "softgnn",
        "--prefix", "./data/BGP_data/as",
        "--gpu", str(gpu),
        "--seed", str(seed),
        "--epochs", str(OPTIMAL_PARAMS["epochs"]),
        "--patience", str(OPTIMAL_PARAMS["patience"]),
        "--num-hidden", str(OPTIMAL_PARAMS["num_hidden"]),
        "--num-layers", str(OPTIMAL_PARAMS["num_layers"]),
        "--lr", str(OPTIMAL_PARAMS["lr"]),
        "--in-drop", str(OPTIMAL_PARAMS["in_drop"]),
        "--softgnn-feat-drop", str(OPTIMAL_PARAMS["softgnn_feat_drop"]),
        "--softgnn-attn-dim", str(OPTIMAL_PARAMS["softgnn_attn_dim"]),
        "--smooth-lambda", str(OPTIMAL_PARAMS["smooth_lambda"]),
        "--batch_size", str(OPTIMAL_PARAMS["batch_size"]),
        "--class-weighted",
    ]
    return cmd


def parse_output(output_text):
    """Parse test accuracy from training output"""
    lines = output_text.split('\n')
    for line in reversed(lines):
        if "The test" in line and "Acc:" in line:
            parts = line.split("Acc:")
            if len(parts) == 2:
                try:
                    acc = float(parts[1].strip())
                    return acc
                except ValueError:
                    continue
    return None


def run_experiment(seed, run_number, total_runs, gpu=0, verbose=True):
    """Run a single experiment with a given seed"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"[{run_number}/{total_runs}] Running SoftGNN on BGP with seed={seed}")
        print(f"{'='*80}")
        print(f"Hyperparameters: {json.dumps(OPTIMAL_PARAMS, indent=2)}")
        print(f"BGP-specific flags: --class-weighted")
        print(f"{'='*80}\n")

    cmd = build_command(seed, gpu=gpu)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        test_acc = parse_output(result.stdout)

        if test_acc is None:
            print(f"WARNING: Could not parse test accuracy from output!")
            print("Last 20 lines of output:")
            print('\n'.join(result.stdout.split('\n')[-20:]))
            return None

        if verbose:
            print(f"\n✓ Seed {seed}: Test Accuracy = {test_acc:.4f}")

        return test_acc

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running seed {seed}:")
        print(f"Return code: {e.returncode}")
        print(f"Error output:\n{e.stderr[-1000:]}")
        return None


def main():
    """Main experiment runner"""
    print("\n" + "="*80)
    print("SoftGNN on BGP Dataset - 10 Random Seed Experiments")
    print("="*80)
    print(f"Model: SoftGNN (Soft Label Graph Neural Network)")
    print(f"Dataset: BGP (Border Gateway Protocol)")
    print(f"Number of seeds: {len(SEEDS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"BGP-specific: --class-weighted (imbalanced)")
    print("="*80 + "\n")

    # Check GPU availability
    gpu = int(input("Enter GPU ID to use (-1 for CPU, 0 for GPU:0): ").strip() or "0")

    # Store results
    results = []

    # Run all experiments
    for i, seed in enumerate(SEEDS, 1):
        test_acc = run_experiment(seed, i, len(SEEDS), gpu=gpu, verbose=True)

        results.append({
            "seed": seed,
            "test_accuracy": test_acc
        })

    # Compute statistics
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80 + "\n")

    valid_results = [r for r in results if r["test_accuracy"] is not None]
    accuracies = [r["test_accuracy"] for r in valid_results]

    if len(accuracies) > 0:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)

        print(f"Successfully completed: {len(valid_results)}/{len(SEEDS)} experiments\n")
        print(f"Test Accuracy Statistics:")
        print(f"  Mean: {mean_acc:.4f}")
        print(f"  Std:  {std_acc:.4f}")
        print(f"  Min:  {min_acc:.4f}")
        print(f"  Max:  {max_acc:.4f}")
        print(f"\n  Result: {mean_acc:.4f} ± {std_acc:.4f}")

        print("\n" + "-"*80)
        print("Individual Results:")
        print("-"*80)

        sorted_results = sorted(valid_results, key=lambda x: x["test_accuracy"], reverse=True)

        for i, result in enumerate(sorted_results, 1):
            acc = result["test_accuracy"]
            seed = result["seed"]
            print(f"  {i:2d}. Seed {seed:5d} | Acc: {acc:.4f}")

        # Save results to JSON
        output_file = f"softgnn_bgp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "model": "softgnn",
                "dataset": "bgp",
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": OPTIMAL_PARAMS,
                "bgp_flags": ["--class-weighted"],
                "summary": {
                    "mean_accuracy": float(mean_acc),
                    "std_accuracy": float(std_acc),
                    "min_accuracy": float(min_acc),
                    "max_accuracy": float(max_acc),
                    "num_experiments": len(valid_results),
                    "num_failed": len(SEEDS) - len(valid_results)
                },
                "results": results
            }, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    else:
        print("ERROR: No successful experiments!")

    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
