"""
run_seeds.py
============
Runs baseline experiment 3 times with different seeds.
Reports mean ± std for each defense.
Saves to results/seeds.csv.

Why multiple seeds?
  Single run results can be lucky or unlucky
  Mean ± std across 3 seeds proves results are consistent
  Required for any publishable research

Run:
  python experiments/run_seeds.py
  python experiments/run_seeds.py --rounds 20
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import argparse
import numpy as np
from core.fl_runner import run_one_experiment

SEEDS   = [42, 123, 456]
DEFENSES = ['none', 'feature', 'subject_aware', 'dp', 'subject_aware_delta']


def run(args):
    base_config = {
        'rounds'             : args.rounds,
        'participation_rate' : args.participation_rate,
        'alpha'              : args.alpha,
        'local_epochs'       : args.local_epochs,
        'lr'                 : args.lr,
        'batch_size'         : args.batch_size,
        'noise_scale'        : args.noise_scale,
        'clip_threshold'     : args.clip_threshold,
        'noise_multiplier'   : args.noise_multiplier,
        'warmup_rounds'      : args.warmup_rounds,
        'track_per_round'    : False,
    }

    # Collect results per defense per seed
    # shape: defense → list of (activity_acc, worst_attack_acc, {clf: acc}) per seed
    all_results = {d: [] for d in DEFENSES}
    csv_rows    = []

    total = len(SEEDS) * len(DEFENSES)
    done  = 0

        # NOTE: subject_aware_delta shows high variance across seeds
    # (observed range: 17.8% to 35.6% attack accuracy across 3 seeds).
    # 3 seeds is minimum. For publication, use >= 5 seeds.
    # Variance comes from random participation in warmup not covering
    # all subjects — fixed in fl_runner.py with min_subjects guard.
    print(f"\n{'='*55}")
    print(f"  Multi-Seed Experiment ({len(SEEDS)} seeds × {len(DEFENSES)} defenses)")
    print(f"{'='*55}\n")

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        for defense in DEFENSES:
            done += 1
            print(f"  [{done}/{total}] defense={defense}...")

            config = {**base_config, 'seed': seed, 'defense_type': defense}
            result = run_one_experiment(config)

            act     = result['final_activity_acc']
            atk     = result['final_attack_acc']
            per_clf = result['attack_acc_per_clf']
            all_results[defense].append((act, atk, per_clf))

            print(
                f"    Activity: {act:.2f}%  Attack (worst): {atk:.2f}%  "
                f"[LR {per_clf.get('lr', 0):.1f}%  "
                f"RF {per_clf.get('rf', 0):.1f}%  "
                f"MLP {per_clf.get('mlp', 0):.1f}%]"
            )

            csv_rows.append({
                'seed'           : seed,
                'defense'        : defense,
                'activity_acc'   : round(act, 4),
                'attack_acc'     : round(atk, 4),
                'attack_acc_lr'  : round(per_clf.get('lr',  0), 4),
                'attack_acc_rf'  : round(per_clf.get('rf',  0), 4),
                'attack_acc_mlp' : round(per_clf.get('mlp', 0), 4),
            })

    # Save raw results to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/seeds.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print mean ± std summary
    print(f"\n{'='*95}")
    print(f"  MULTI-SEED RESULTS — Mean ± Std over {len(SEEDS)} seeds")
    print(f"{'='*95}")
    print(f"  {'Defense':<22} {'Activity':>16} {'LR':>16} {'RF':>16} {'MLP':>16} {'Worst':>16}")
    print(f"  {'-'*87}")

    for defense in DEFENSES:
        acts = [r[0] for r in all_results[defense]]
        atks = [r[1] for r in all_results[defense]]
        lrs  = [r[2].get('lr',  0) for r in all_results[defense]]
        rfs  = [r[2].get('rf',  0) for r in all_results[defense]]
        mlps = [r[2].get('mlp', 0) for r in all_results[defense]]
        tag  = " ←" if defense == 'subject_aware_delta' else ""
        print(
            f"  {defense:<22}"
            f"  {np.mean(acts):>6.2f}±{np.std(acts):.2f}%"
            f"  {np.mean(lrs):>6.2f}±{np.std(lrs):.2f}%"
            f"  {np.mean(rfs):>6.2f}±{np.std(rfs):.2f}%"
            f"  {np.mean(mlps):>6.2f}±{np.std(mlps):.2f}%"
            f"  {np.mean(atks):>6.2f}±{np.std(atks):.2f}%"
            f"{tag}"
        )

    print(f"{'='*95}")
    print(f"\n  Raw results saved to: {csv_path}")
    print(f"  Use mean ± std when writing up results\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds",             type=int,   default=15)
    parser.add_argument("--participation_rate", type=float, default=0.5)
    parser.add_argument("--alpha",              type=float, default=0.5)
    parser.add_argument("--noise_scale",        type=float, default=2.0)
    parser.add_argument("--clip_threshold",     type=float, default=1.0)
    parser.add_argument("--noise_multiplier",   type=float, default=0.05)
    parser.add_argument("--warmup_rounds",      type=int,   default=5)
    parser.add_argument("--local_epochs",       type=int,   default=3)
    parser.add_argument("--lr",                 type=float, default=0.01)
    parser.add_argument("--batch_size",         type=int,   default=32)
    args = parser.parse_args()
    run(args)