"""
run_per_round.py
================
Tracks attack accuracy every few rounds for each defense.
Saves to results/per_round.csv.

This answers:
  - Does attack get stronger as training progresses?
  - At what round does the attacker become dangerous?
  - Do defenses degrade over time?

Run:
  python experiments/run_per_round.py
  python experiments/run_per_round.py --track_every 2
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import argparse
from core.fl_runner import run_one_experiment

DEFENSES = ['none', 'feature', 'subject_aware', 'dp', 'subject_aware_delta']


def run(args):
    base_config = {
        'seed'               : 42,
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
        'track_per_round'    : True,          # ← enabled
        'track_every'        : args.track_every,
    }

    csv_rows = []

    print(f"\n{'='*55}")
    print(f"  Per-Round Attack Tracking")
    print(f"  Rounds: {args.rounds}  Track every: {args.track_every} rounds")
    print(f"{'='*55}\n")

    for defense in DEFENSES:
        print(f"Running defense: {defense}...")
        config = {**base_config, 'defense_type': defense}
        result = run_one_experiment(config)

        # per_round_attack is list of (round_number, attack_acc)
        per_round = result['per_round_attack']
        per_round_acc = result['per_round_acc']

        print(f"  Per-round attack accuracy:")
        for rnd, atk_acc, per_clf in per_round:
            act_acc = per_round_acc[rnd - 1]   # round is 1-indexed
            print(
                f"    Round {rnd:>3} → Activity: {act_acc:.2f}%  "
                f"Attack (worst): {atk_acc:.2f}%  "
                f"[LR {per_clf.get('lr', 0):.1f}%  "
                f"RF {per_clf.get('rf', 0):.1f}%  "
                f"MLP {per_clf.get('mlp', 0):.1f}%]"
            )

            csv_rows.append({
                'defense'               : defense,
                'round'                 : rnd,
                'activity_acc'          : round(act_acc, 4),
                'attack_acc'            : round(atk_acc, 4),
                'attack_acc_lr'         : round(per_clf.get('lr',  0), 4),
                'attack_acc_rf'         : round(per_clf.get('rf',  0), 4),
                'attack_acc_mlp'        : round(per_clf.get('mlp', 0), 4),
                'privacy_gain_vs_random': round(atk_acc / (100.0 / 30), 4),
                'alpha'                 : args.alpha,
                'seed'                  : 42,
            })
        print()

    # Save to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/per_round.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"{'='*55}")
    print(f"  Saved to: {csv_path}")
    print(f"  Use this CSV to plot attack accuracy over rounds")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds",             type=int,   default=20)
    parser.add_argument("--track_every",        type=int,   default=5,
                        help="Track attack accuracy every N rounds")
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