"""
run_baseline.py
===============
Runs all 5 defenses once with fixed config.
Saves results to results/baseline.csv.

Run:
  python experiments/run_baseline.py
  python experiments/run_baseline.py --rounds 20 --participation_rate 1.0
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
        'track_per_round'    : False,
    }

    results      = {}
    csv_rows     = []

    print(f"\n{'='*55}")
    print(f"  Baseline Experiment — All 5 Defenses")
    print(f"{'='*55}\n")

    for defense in DEFENSES:
        print(f"Running defense: {defense}...")
        config = {**base_config, 'defense_type': defense}
        result = run_one_experiment(config)
        results[defense] = result

        per_clf = result['attack_acc_per_clf']
        print(
            f"  Activity: {result['final_activity_acc']:.2f}%  "
            f"Attack (worst): {result['final_attack_acc']:.2f}%  "
            f"[LR {per_clf.get('lr', 0):.1f}%  "
            f"RF {per_clf.get('rf', 0):.1f}%  "
            f"MLP {per_clf.get('mlp', 0):.1f}%]\n"
        )

        # privacy_gain is computed after all defenses run (needs baseline_attack).
        # Store a placeholder and fill it in after the loop.
        csv_rows.append({
            'defense'               : defense,
            'activity_acc'          : round(result['final_activity_acc'], 4),
            'attack_acc'            : round(result['final_attack_acc'], 4),
            'attack_acc_lr'         : round(per_clf.get('lr',  0), 4),
            'attack_acc_rf'         : round(per_clf.get('rf',  0), 4),
            'attack_acc_mlp'        : round(per_clf.get('mlp', 0), 4),
            'privacy_gain'          : None,   # filled below
            'privacy_gain_vs_random': None,   # filled below
            'rounds'                : args.rounds,
            'noise_scale'           : args.noise_scale,
            'noise_multiplier'      : args.noise_multiplier,
            'alpha'                 : args.alpha,
            'seed'                  : 42,
        })

    # Fill in privacy_gain columns — needs no-defense baseline, so computed after loop
    baseline_attack_for_gain = results['none']['final_attack_acc']
    random_chance            = 100.0 / 30   # 30 subjects => random baseline
    for row in csv_rows:
        atk = row['attack_acc']
        row['privacy_gain']           = round(baseline_attack_for_gain - atk, 4)
        row['privacy_gain_vs_random'] = round(atk / random_chance, 4)

    # Save to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/baseline.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print final table
    baseline_attack = results['none']['final_attack_acc']
    random_baseline = 100.0 / 30

    print(f"\n{'='*85}")
    print(f"  FINAL RESULTS")
    print(f"{'='*85}")
    print(f"  {'Defense':<22} {'Activity':>9} {'LR':>8} {'RF':>8} {'MLP':>8} {'Worst':>8} {'PrivGain':>10}")
    print(f"  {'-'*77}")
    for defense in DEFENSES:
        r       = results[defense]
        per_clf = r['attack_acc_per_clf']
        gain    = baseline_attack - r['final_attack_acc']
        tag     = " ←" if defense == 'subject_aware_delta' else ""
        print(
            f"  {defense:<22}"
            f"{r['final_activity_acc']:>8.1f}%"
            f"{per_clf.get('lr',  0):>7.1f}%"
            f"{per_clf.get('rf',  0):>7.1f}%"
            f"{per_clf.get('mlp', 0):>7.1f}%"
            f"{r['final_attack_acc']:>7.1f}%"
            f"{gain:>+9.1f}%"
            f"{tag}"
        )
    print(f"  {'Random baseline':<22} {'—':>9} {'—':>8} {'—':>8} {'—':>8} {random_baseline:>7.1f}%")
    print(f"{'='*85}")
    print(f"\n  Results saved to: {csv_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds",             type=int,   default=20)
    parser.add_argument("--participation_rate", type=float, default=1.0)
    parser.add_argument("--alpha",              type=float, default=0.5)
    parser.add_argument("--noise_scale",        type=float, default=2.0)
    parser.add_argument("--clip_threshold",     type=float, default=1.0)
    parser.add_argument("--noise_multiplier",   type=float, default=0.05)
    parser.add_argument("--warmup_rounds",      type=int,   default=5)
    parser.add_argument("--local_epochs",       type=int,   default=3)
    parser.add_argument("--lr",                 type=float, default=0.01)
    parser.add_argument("--batch_size",         type=int,   default=32)
    parser.add_argument("--track_every",        type=int,   default=5)
    args = parser.parse_args()
    run(args)