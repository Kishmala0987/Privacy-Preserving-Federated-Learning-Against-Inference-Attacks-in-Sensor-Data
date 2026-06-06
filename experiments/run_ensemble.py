"""
run_ensemble.py
===============
Sweep K (number of ensemble groups) and defense combinations.

Experiments run:
  1. K sweep (no defense)         : K ∈ {1, 2, 3, 5}
     K=1 degenerates to standard FedAvg — sanity check.

  2. Defense sweep at best K      : all 5 defenses at K=3
     Compares each defense under ensemble vs standard FedAvg.

  3. Combination sweep            : best defense × best K
     subject_aware_delta at K ∈ {1, 2, 3, 5}

Results saved to:
  results/ensemble_k_sweep.csv
  results/ensemble_defense_sweep.csv
  results/ensemble_combination.csv

Usage:
  cd FL_Project3_Ensemble
  python experiments/run_ensemble.py
  python experiments/run_ensemble.py --rounds 20 --best_k 5
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import argparse
import time

from core.ensemble_runner import run_ensemble_experiment


RANDOM_CHANCE = 100.0 / 30   # 30 subjects


def make_base_config(args):
    return {
        'seed'              : 42,
        'rounds'            : args.rounds,
        'participation_rate': 0.7,
        'alpha'             : 0.5,
        'local_epochs'      : 3,
        'lr'                : 0.01,
        'batch_size'        : 32,
        'noise_scale'       : 2.0,
        'clip_threshold'    : 1.0,
        'noise_multiplier'  : 0.05,
        'warmup_rounds'     : 5,
        'sensitive_top_k'   : 50,
        'ensemble_inference': 'avg_proba',
        'track_per_round'   : False,
    }


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def row_from_result(result, extra):
    """Build one CSV row from a result dict + extra metadata."""
    r = result
    row = {
        'defense'               : extra.get('defense', 'none'),
        'n_groups'              : r['n_groups'],
        'activity_acc'          : round(r['final_activity_acc'], 4),
        'attack_acc'            : round(r['final_attack_acc'], 4),
        'attack_acc_lr'         : round(r['attack_acc_per_clf'].get('lr',  0), 4),
        'attack_acc_rf'         : round(r['attack_acc_per_clf'].get('rf',  0), 4),
        'attack_acc_mlp'        : round(r['attack_acc_per_clf'].get('mlp', 0), 4),
        'privacy_gain_vs_random': r['privacy_gain_vs_random'],
        'ensemble_gain'         : r['ensemble_gain'],
        'group_sizes'           : str(r['group_sizes']),
        'group_activity_accs'   : str([round(a, 2) for a in r['group_activity_accs']]),
        'rounds'                : extra.get('rounds', '?'),
        'seed'                  : extra.get('seed', 42),
    }
    return row


# ─────────────────────────────────────────
# Experiment 1 — K sweep (no defense)
# ─────────────────────────────────────────

def run_k_sweep(args):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 1: K sweep (defense=none)")
    print(f"{'='*60}")

    k_values = [1, 2, 3, 5]
    rows     = []

    for k in k_values:
        print(f"\n  ── K={k} ──")
        cfg    = make_base_config(args)
        cfg.update({'n_groups': k, 'defense_type': 'none'})
        t0     = time.time()
        result = run_ensemble_experiment(cfg)
        elapsed = time.time() - t0
        row    = row_from_result(result, {'defense': 'none', 'rounds': args.rounds, 'seed': 42})
        rows.append(row)
        print(f"  K={k} | Activity: {result['final_activity_acc']:.1f}% | "
              f"Attack: {result['final_attack_acc']:.1f}% | "
              f"EnsGain: {result['ensemble_gain']:+.1f}pp | "
              f"Time: {elapsed:.0f}s")

    save_csv(rows, 'results/ensemble_k_sweep.csv')

    # Print summary table
    print(f"\n  {'K':<5} {'Activity':>9} {'Attack':>8} {'vs Random':>11} {'Ens Gain':>10}")
    print(f"  {'-'*47}")
    for row in rows:
        print(f"  {row['n_groups']:<5} {row['activity_acc']:>8.2f}% "
              f"{row['attack_acc']:>7.2f}% "
              f"{row['privacy_gain_vs_random']:>9.1f}x "
              f"{row['ensemble_gain']:>+9.2f}pp")

    return rows


# ─────────────────────────────────────────
# Experiment 2 — Defense sweep at best K
# ─────────────────────────────────────────

def run_defense_sweep(args):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 2: Defense sweep at K={args.best_k}")
    print(f"{'='*60}")

    defenses = ['none', 'feature', 'subject_aware', 'dp', 'subject_aware_delta']
    rows     = []

    for defense in defenses:
        print(f"\n  ── defense={defense} ──")
        cfg = make_base_config(args)
        cfg.update({'n_groups': args.best_k, 'defense_type': defense})
        t0     = time.time()
        result = run_ensemble_experiment(cfg)
        elapsed = time.time() - t0
        row    = row_from_result(result, {'defense': defense, 'rounds': args.rounds, 'seed': 42})
        rows.append(row)
        print(f"  {defense:<22} | Activity: {result['final_activity_acc']:.1f}% | "
              f"Attack: {result['final_attack_acc']:.1f}% | "
              f"Time: {elapsed:.0f}s")

    save_csv(rows, 'results/ensemble_defense_sweep.csv')

    # Print summary table
    print(f"\n  {'Defense':<24} {'Activity':>9} {'Attack':>8} {'vs Random':>11} {'Ens Gain':>10}")
    print(f"  {'-'*65}")
    for row in rows:
        print(f"  {row['defense']:<24} {row['activity_acc']:>8.2f}% "
              f"{row['attack_acc']:>7.2f}% "
              f"{row['privacy_gain_vs_random']:>9.1f}x "
              f"{row['ensemble_gain']:>+9.2f}pp")

    return rows


# ─────────────────────────────────────────
# Experiment 3 — Combination: best defense × K sweep
# ─────────────────────────────────────────

def run_combination_sweep(args):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 3: subject_aware_delta × K sweep")
    print(f"{'='*60}")

    k_values = [1, 2, 3, 5]
    rows     = []

    for k in k_values:
        print(f"\n  ── K={k}, defense=subject_aware_delta ──")
        cfg = make_base_config(args)
        cfg.update({'n_groups': k, 'defense_type': 'subject_aware_delta'})
        t0     = time.time()
        result = run_ensemble_experiment(cfg)
        elapsed = time.time() - t0
        row    = row_from_result(
            result,
            {'defense': 'subject_aware_delta', 'rounds': args.rounds, 'seed': 42}
        )
        rows.append(row)
        print(f"  K={k} | Activity: {result['final_activity_acc']:.1f}% | "
              f"Attack: {result['final_attack_acc']:.1f}% | "
              f"EnsGain: {result['ensemble_gain']:+.1f}pp | "
              f"Time: {elapsed:.0f}s")

    save_csv(rows, 'results/ensemble_combination.csv')

    print(f"\n  {'K':<5} {'Activity':>9} {'Attack':>8} {'vs Random':>11} {'Ens Gain':>10}")
    print(f"  {'-'*47}")
    for row in rows:
        print(f"  {row['n_groups']:<5} {row['activity_acc']:>8.2f}% "
              f"{row['attack_acc']:>7.2f}% "
              f"{row['privacy_gain_vs_random']:>9.1f}x "
              f"{row['ensemble_gain']:>+9.2f}pp")

    return rows


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Ensemble FL experiments')
    parser.add_argument('--rounds',      type=int,   default=15,
                        help='Communication rounds (default 15, use 20 for paper results)')
    parser.add_argument('--best_k',      type=int,   default=3,
                        help='K to use for defense sweep (Experiment 2)')
    parser.add_argument('--exp',         type=str,   default='all',
                        choices=['all', 'k_sweep', 'defense_sweep', 'combination'],
                        help='Which experiment to run')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Ensemble FL Experiments")
    print(f"  rounds={args.rounds} | best_k={args.best_k} | exp={args.exp}")
    print(f"{'='*60}")

    if args.exp in ('all', 'k_sweep'):
        run_k_sweep(args)

    if args.exp in ('all', 'defense_sweep'):
        run_defense_sweep(args)

    if args.exp in ('all', 'combination'):
        run_combination_sweep(args)

    print(f"\n{'='*60}")
    print(f"  All results saved to results/")
    print(f"  Run plot_results.py to visualize")
    print(f"{'='*60}\n")
