"""
run_ablation.py
===============
Systematic sweep of noise values for each defense.
Saves full sweep to results/ablation.csv.

What it sweeps:
  feature/subject_aware → noise_scale values
  dp                    → noise_multiplier values
  subject_aware_delta   → sensitive_multiplier passed via noise_scale;
                          client.py uses noise_scale as sensitive_multiplier
                          for this defense so the sweep is live

Each row in CSV = one (defense, noise_value) combination.
Use this CSV to plot the privacy-utility tradeoff curve later.

Run:
  python experiments/run_ablation.py
  python experiments/run_ablation.py --rounds 15
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import argparse
from core.fl_runner import run_one_experiment

# Sweep values for each defense type
SWEEP = {
    'none'                : [0],           # no noise parameter, run once
    'feature'             : [0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
    'subject_aware'       : [0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
    'dp'                  : [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
    'subject_aware_delta' : [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
}


def run(args):
    base_config = {
        'seed'               : 42,
        'rounds'             : args.rounds,
        'participation_rate' : args.participation_rate,
        'alpha'              : args.alpha,
        'local_epochs'       : args.local_epochs,
        'lr'                 : args.lr,
        'batch_size'         : args.batch_size,
        'clip_threshold'     : 1.0,
        'warmup_rounds'      : args.warmup_rounds,
        'track_per_round'    : False,
    }

    csv_rows = []
    total    = sum(len(v) for v in SWEEP.values())
    done     = 0

    print(f"\n{'='*55}")
    print(f"  Ablation Sweep — {total} experiments total")
    print(f"{'='*55}\n")

    for defense, noise_values in SWEEP.items():
        for noise_val in noise_values:
            done += 1
            print(f"[{done}/{total}] defense={defense}  noise={noise_val}...")

            # Set the right noise parameter per defense
            if defense == 'dp':
                config = {
                    **base_config,
                    'defense_type'    : defense,
                    'noise_multiplier': noise_val,
                    'noise_scale'     : 0.0,
                }
            elif defense == 'none':
                config = {
                    **base_config,
                    'defense_type'    : defense,
                    'noise_scale'     : 0.0,
                    'noise_multiplier': 0.0,
                }
            else:
                config = {
                    **base_config,
                    'defense_type'    : defense,
                    'noise_scale'     : noise_val,
                    'noise_multiplier': 0.0,
                }

            result  = run_one_experiment(config)
            per_clf = result['attack_acc_per_clf']

            print(
                f"   Activity: {result['final_activity_acc']:.2f}%  "
                f"Attack (worst): {result['final_attack_acc']:.2f}%  "
                f"[LR {per_clf.get('lr', 0):.1f}%  "
                f"RF {per_clf.get('rf', 0):.1f}%  "
                f"MLP {per_clf.get('mlp', 0):.1f}%]"
            )

            csv_rows.append({
                'defense'         : defense,
                'noise_value'     : noise_val,
                'activity_acc'    : round(result['final_activity_acc'], 4),
                'attack_acc'      : round(result['final_attack_acc'], 4),
                'attack_acc_lr'   : round(per_clf.get('lr',  0), 4),
                'attack_acc_rf'   : round(per_clf.get('rf',  0), 4),
                'attack_acc_mlp'  : round(per_clf.get('mlp', 0), 4),
                'rounds'          : args.rounds,
                'alpha'           : args.alpha,
                'seed'            : 42,
            })

    # Save to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/ablation.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*55}")
    print(f"  Ablation complete. Saved to: {csv_path}")
    print(f"  Use this CSV to plot privacy-utility curves")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds",             type=int,   default=15)
    parser.add_argument("--participation_rate", type=float, default=0.5)
    parser.add_argument("--alpha",              type=float, default=0.5)
    parser.add_argument("--warmup_rounds",      type=int,   default=5)
    parser.add_argument("--local_epochs",       type=int,   default=3)
    parser.add_argument("--lr",                 type=float, default=0.01)
    parser.add_argument("--batch_size",         type=int,   default=32)
    args = parser.parse_args()
    run(args)