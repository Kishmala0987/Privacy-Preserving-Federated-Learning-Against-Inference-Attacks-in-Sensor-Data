"""
plot_results.py
===============
Reads all CSV results and generates publication-ready plots.
Saves all figures to plots/ folder.

Plots generated:
  1. baseline_comparison.png      — grouped bars: activity vs attack per defense
  2. baseline_per_clf.png         — per-classifier attack breakdown (LR / RF / MLP)
  3. ablation_curves.png          — activity & attack vs noise strength
  4. privacy_utility_scatter.png  — scatter: attack vs activity (tradeoff curve)
  5. privacy_gain_vs_random.png   — how many times above random each defense is
  6. seeds_comparison.png         — mean ± std across 3 seeds
  7. per_round_tracking.png       — activity & attack accuracy over rounds

Run:
  python experiments/plot_results.py

Requires:
  pip install matplotlib pandas
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs('plots', exist_ok=True)

# ── Color scheme — one color per defense ──────────────────
COLORS = {
    'none'                : '#e74c3c',   # red    — danger/no defense
    'feature'             : '#3498db',   # blue
    'subject_aware'       : '#e67e22',   # orange
    'dp'                  : '#2ecc71',   # green  — strongest DP defense
    'subject_aware_delta' : '#9b59b6',   # purple — novel contribution
}

LABELS = {
    'none'                : 'No Defense',
    'feature'             : 'Feature Noise',
    'subject_aware'       : 'Subject-Aware (Feature)',
    'dp'                  : 'DP (Delta)',
    'subject_aware_delta' : 'Subject-Aware (Delta) ★',
}

DEFENSES = list(COLORS.keys())

RANDOM_BASELINE = 100.0 / 30   # 3.33% — 30 subjects


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _bar_labels(ax, bars, fmt='{:.1f}%', pad=0.5, fontsize=7.5):
    """Add value labels on top of bars."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad,
            fmt.format(h),
            ha='center', va='bottom', fontsize=fontsize,
        )


def _defense_legend(ax, defenses, fontsize=9):
    patches = [
        mpatches.Patch(color=COLORS[d], label=LABELS.get(d, d))
        for d in defenses
    ]
    ax.legend(handles=patches, fontsize=fontsize)


# ─────────────────────────────────────────────────────────
# Plot 1 — Baseline grouped bar chart
# ─────────────────────────────────────────────────────────

def plot_baseline():
    path = 'results/baseline.csv'
    if not os.path.exists(path):
        print(f"  Skipping baseline plot — {path} not found")
        return

    df = pd.read_csv(path)
    defenses     = df['defense'].tolist()
    activity_acc = df['activity_acc'].tolist()
    attack_acc   = df['attack_acc'].tolist()

    # privacy_gain column — show reduction vs no-defense
    has_gain = 'privacy_gain' in df.columns
    if has_gain:
        priv_gain = df['privacy_gain'].tolist()

    x     = np.arange(len(defenses))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x - width / 2, activity_acc, width,
                   label='Activity Accuracy',
                   color=[COLORS[d] for d in defenses],
                   alpha=0.85)
    bars2 = ax.bar(x + width / 2, attack_acc, width,
                   label='Attack Accuracy (worst-case)',
                   color=[COLORS[d] for d in defenses],
                   alpha=0.45, hatch='//')

    ax.axhline(y=RANDOM_BASELINE, color='black', linestyle='--',
               linewidth=1.2, label=f'Random Baseline ({RANDOM_BASELINE:.1f}%)')

    _bar_labels(ax, bars1)
    _bar_labels(ax, bars2)

    # Annotate privacy_gain on each attack bar
    if has_gain:
        for bar, gain in zip(bars2, priv_gain):
            if gain > 0.1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3.5,
                    f'−{gain:.1f}pp',
                    ha='center', va='bottom', fontsize=7,
                    color='darkgreen', fontstyle='italic',
                )

    ax.set_xlabel('Defense Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Privacy-Utility Tradeoff — All Defenses\n'
                 '(solid = activity accuracy, hatched = worst-case attack accuracy)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(d, d) for d in defenses], rotation=15, ha='right')
    ax.set_ylim(0, 118)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/baseline_comparison.png', dpi=150)
    plt.close()
    print("  Saved: plots/baseline_comparison.png")


# ─────────────────────────────────────────────────────────
# Plot 2 — Per-classifier attack breakdown (baseline)
# ─────────────────────────────────────────────────────────

def plot_baseline_per_clf():
    """Per-classifier attack breakdown — reads directly from baseline.csv."""
    path = 'results/baseline.csv'
    if not os.path.exists(path):
        print(f"  Skipping per-clf plot — {path} not found")
        return

    df = pd.read_csv(path)
    needed = {'attack_acc_lr', 'attack_acc_rf', 'attack_acc_mlp'}
    if not needed.issubset(df.columns):
        print("  Skipping per-clf plot — re-run run_baseline.py to generate per-classifier columns")
        return

    defenses = [d for d in DEFENSES if d in df['defense'].values]
    df = df.set_index('defense').loc[defenses].reset_index()

    lr_vals  = df['attack_acc_lr'].tolist()
    rf_vals  = df['attack_acc_rf'].tolist()
    mlp_vals = df['attack_acc_mlp'].tolist()
    worst    = df['attack_acc'].tolist()

    x     = np.arange(len(defenses))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    # Three classifiers side-by-side, same color per defense, different alpha/hatch
    b1 = ax.bar(x - width,      lr_vals,  width, label='LR  (weak)',   alpha=0.55,
                color=[COLORS[d] for d in defenses], hatch='')
    b2 = ax.bar(x,              rf_vals,  width, label='RF  (medium)', alpha=0.75,
                color=[COLORS[d] for d in defenses], hatch='..')
    b3 = ax.bar(x + width,      mlp_vals, width, label='MLP (strong)', alpha=0.95,
                color=[COLORS[d] for d in defenses], hatch='//')

    # Worst-case marker as a horizontal tick on each group
    for i, w in enumerate(worst):
        ax.plot([x[i] - width * 1.6, x[i] + width * 1.6], [w, w],
                color=COLORS[defenses[i]], linewidth=2.5, linestyle='-',
                solid_capstyle='round', zorder=5)
        ax.text(x[i] + width * 1.7, w, f'{w:.1f}%',
                va='center', fontsize=7.5, color=COLORS[defenses[i]], fontweight='bold')

    ax.axhline(y=RANDOM_BASELINE, color='black', linestyle='--',
               linewidth=1.2, label=f'Random ({RANDOM_BASELINE:.1f}%)')

    # Value labels on bars
    for bars in (b1, b2, b3):
        _bar_labels(ax, bars, pad=0.3, fontsize=7)

    ax.set_xlabel('Defense Method', fontsize=12)
    ax.set_ylabel('Attack Accuracy (%)', fontsize=12)
    ax.set_title('Per-Classifier Attack Accuracy at Baseline Noise Setting\n'
                 '(bar color = defense · shade/hatch = classifier · horizontal line = worst-case)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(d, d) for d in defenses], rotation=15, ha='right')
    ax.set_ylim(0, 118)

    # Legend: classifier styles
    clf_leg = ax.legend(fontsize=9, loc='upper right', title='Classifier strength')

    # Legend: defense colors
    color_patches = [mpatches.Patch(color=COLORS[d], label=LABELS.get(d, d))
                     for d in defenses]
    ax.add_artist(clf_leg)
    ax.legend(handles=color_patches, fontsize=8, loc='upper left', title='Defense')

    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, 0.01,
            'Horizontal line = worst-case across classifiers — the number reported in the paper',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('plots/baseline_per_clf.png', dpi=150)
    plt.close()
    print("  Saved: plots/baseline_per_clf.png")


# ─────────────────────────────────────────────────────────
# Plot 3 — Ablation: privacy-utility tradeoff curves
# ─────────────────────────────────────────────────────────

def plot_ablation():
    path = 'results/ablation.csv'
    if not os.path.exists(path):
        print(f"  Skipping ablation plot — {path} not found")
        return

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Activity vs noise value ──
    ax1 = axes[0]
    for defense in DEFENSES:
        sub = df[df['defense'] == defense].sort_values('noise_value')
        # Drop blank rows (subject_aware_delta needs rerun with fixed code)
        sub = sub[pd.to_numeric(sub['activity_acc'], errors='coerce').notna()]
        if sub.empty:
            continue
        ax1.plot(sub['noise_value'], sub['activity_acc'].astype(float),
                 marker='o', color=COLORS[defense],
                 label=LABELS.get(defense, defense), linewidth=2)

    ax1.set_xlabel('Noise Value / Sensitive Multiplier', fontsize=12)
    ax1.set_ylabel('Activity Accuracy (%)', fontsize=12)
    ax1.set_title('Activity Accuracy vs Noise Strength', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Right: Attack vs noise value ──
    ax2 = axes[1]
    for defense in DEFENSES:
        sub = df[df['defense'] == defense].sort_values('noise_value')
        sub = sub[pd.to_numeric(sub['attack_acc'], errors='coerce').notna()]
        if sub.empty:
            continue
        ax2.plot(sub['noise_value'], sub['attack_acc'].astype(float),
                 marker='o', color=COLORS[defense],
                 label=LABELS.get(defense, defense), linewidth=2)

    ax2.axhline(y=RANDOM_BASELINE, color='black', linestyle='--',
                linewidth=1.2, label=f'Random ({RANDOM_BASELINE:.1f}%)')

    # Annotate DP sweet spot (noise_multiplier ≈ 0.05)
    dp_sub = df[df['defense'] == 'dp'].sort_values('noise_value')
    if not dp_sub.empty:
        sweet = dp_sub[dp_sub['noise_value'] == 0.05]
        if not sweet.empty:
            ax2.annotate(
                'DP sweet spot\n(noise=0.05)',
                xy=(0.05, float(sweet['attack_acc'].iloc[0])),
                xytext=(0.3, 25),
                arrowprops=dict(arrowstyle='->', color='#2ecc71'),
                fontsize=8, color='#2ecc71',
            )

    ax2.set_xlabel('Noise Value', fontsize=12)
    ax2.set_ylabel('Attack Accuracy (%)', fontsize=12)
    ax2.set_title('Attack Accuracy vs Noise Strength', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle('Ablation Study — Effect of Noise Strength\n'
                 '(subject_aware_delta x-axis = sensitive_multiplier; range 1–50)\n'
                 '(★ subject_aware_delta rows blank until re-run with fixed code)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/ablation_curves.png', dpi=150)
    plt.close()
    print("  Saved: plots/ablation_curves.png")


# ─────────────────────────────────────────────────────────
# Plot 4 — Privacy-utility scatter (key research plot)
# ─────────────────────────────────────────────────────────

def plot_privacy_utility_scatter():
    path = 'results/ablation.csv'
    if not os.path.exists(path):
        print(f"  Skipping scatter plot — {path} not found")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(9, 7))

    for defense in DEFENSES:
        sub = df[df['defense'] == defense]
        sub = sub[pd.to_numeric(sub['attack_acc'], errors='coerce').notna()]
        sub = sub[pd.to_numeric(sub['activity_acc'], errors='coerce').notna()]
        if sub.empty:
            continue
        sub = sub.copy()
        sub['attack_acc'] = sub['attack_acc'].astype(float)
        sub['activity_acc'] = sub['activity_acc'].astype(float)
        sc = ax.scatter(sub['attack_acc'], sub['activity_acc'],
                        color=COLORS[defense], s=80, alpha=0.8,
                        label=LABELS.get(defense, defense), zorder=3)
        # Connect points to show the tradeoff curve
        sub_sorted = sub.sort_values('attack_acc')
        ax.plot(sub_sorted['attack_acc'], sub_sorted['activity_acc'],
                color=COLORS[defense], alpha=0.35, linewidth=1.2)

    # Random baseline vertical line
    ax.axvline(x=RANDOM_BASELINE, color='black', linestyle='--',
               linewidth=1, alpha=0.5, label=f'Random baseline ({RANDOM_BASELINE:.1f}%)')

    # Ideal region shading (top-left quadrant)
    ax.axvspan(0, RANDOM_BASELINE * 2, ymin=0.7, alpha=0.08, color='green')
    ax.annotate('Ideal region\n(high utility,\nlow attack)',
                xy=(RANDOM_BASELINE * 2 * 0.5, 88), fontsize=8.5, color='darkgreen',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.35))

    # Annotate subject_aware_delta best point
    sad = df[df['defense'] == 'subject_aware_delta']
    if not sad.empty:
        best = sad.loc[sad['activity_acc'].idxmax()]
        ax.annotate(
            f"★ Best SAD\n({best['activity_acc']:.1f}% / {best['attack_acc']:.1f}%)",
            xy=(best['attack_acc'], best['activity_acc']),
            xytext=(best['attack_acc'] + 8, best['activity_acc'] - 12),
            arrowprops=dict(arrowstyle='->', color='#9b59b6'),
            fontsize=8, color='#9b59b6',
        )

    ax.set_xlabel('Attack Accuracy (%) — lower is better for privacy', fontsize=12)
    ax.set_ylabel('Activity Accuracy (%) — higher is better for utility', fontsize=12)
    ax.set_title('Privacy-Utility Tradeoff Curve\n(closer to top-left = better)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(-2, 107)
    ax.set_ylim(0, 107)

    plt.tight_layout()
    plt.savefig('plots/privacy_utility_scatter.png', dpi=150)
    plt.close()
    print("  Saved: plots/privacy_utility_scatter.png")


# ─────────────────────────────────────────────────────────
# Plot 5 — Privacy gain vs random baseline (new)
# ─────────────────────────────────────────────────────────

def plot_privacy_gain_vs_random():
    """
    Shows attack_acc / random_chance ratio per defense.
    1.0 = attacker is no better than random (perfect privacy).
    Higher = worse privacy.
    Uses baseline.csv if privacy_gain_vs_random column exists,
    else falls back to computing it on the fly.
    """
    path = 'results/baseline.csv'
    if not os.path.exists(path):
        print(f"  Skipping privacy-gain plot — {path} not found")
        return

    df = pd.read_csv(path)

    if 'privacy_gain_vs_random' in df.columns:
        ratios = df['privacy_gain_vs_random'].tolist()
    else:
        ratios = (df['attack_acc'] / RANDOM_BASELINE).tolist()

    defenses = df['defense'].tolist()
    x        = np.arange(len(defenses))
    colors   = [COLORS[d] for d in defenses]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(x, ratios, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)

    # Perfect privacy line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5,
               label='Random baseline (ratio = 1.0 = perfect privacy)')

    # Danger threshold at 2x
    ax.axhline(y=2.0, color='#e74c3c', linestyle=':', linewidth=1.2, alpha=0.7,
               label='2× random — moderate leak threshold')

    # Value labels
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f'{ratio:.1f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Defense Method', fontsize=12)
    ax.set_ylabel('Attack Accuracy / Random Baseline', fontsize=12)
    ax.set_title('Attack Strength Relative to Random Guessing\n'
                 '(1.0 = attacker is no better than random; lower is better)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(d, d) for d in defenses], rotation=15, ha='right')
    ax.set_ylim(0, max(ratios) * 1.2 + 1)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    _defense_legend(ax, defenses, fontsize=8)

    plt.tight_layout()
    plt.savefig('plots/privacy_gain_vs_random.png', dpi=150)
    plt.close()
    print("  Saved: plots/privacy_gain_vs_random.png")


# ─────────────────────────────────────────────────────────
# Plot 6 — Multi-seed bar chart with error bars
# ─────────────────────────────────────────────────────────

def plot_seeds():
    path = 'results/seeds.csv'
    if not os.path.exists(path):
        print(f"  Skipping seeds plot — {path} not found")
        return

    df = pd.read_csv(path)

    summary = df.groupby('defense').agg(
        activity_mean=('activity_acc', 'mean'),
        activity_std=('activity_acc', 'std'),
        attack_mean=('attack_acc', 'mean'),
        attack_std=('attack_acc', 'std'),
    ).reset_index()

    summary['order'] = summary['defense'].map({d: i for i, d in enumerate(DEFENSES)})
    summary = summary.sort_values('order')

    has_ratio = 'privacy_gain_vs_random' in df.columns
    ratio_summary = None
    if has_ratio:
        ratio_summary = df.groupby('defense').agg(
            ratio_mean=('privacy_gain_vs_random', 'mean'),
            ratio_std=('privacy_gain_vs_random', 'std'),
        ).reset_index()
        ratio_summary['order'] = ratio_summary['defense'].map(
            {d: i for i, d in enumerate(DEFENSES)})
        ratio_summary = ratio_summary.sort_values('order')

    # Two panels if ratio column exists, else one
    ncols  = 2 if has_ratio else 1
    fig, axes = plt.subplots(1, ncols, figsize=(12 if has_ratio else 11, 6))
    if ncols == 1:
        axes = [axes]

    ax = axes[0]
    x     = np.arange(len(summary))
    width = 0.35

    ax.bar(x - width / 2, summary['activity_mean'], width,
           yerr=summary['activity_std'], capsize=4,
           color=[COLORS[d] for d in summary['defense']],
           alpha=0.85, label='Activity Accuracy')
    ax.bar(x + width / 2, summary['attack_mean'], width,
           yerr=summary['attack_std'], capsize=4,
           color=[COLORS[d] for d in summary['defense']],
           alpha=0.45, hatch='//', label='Attack Accuracy (worst-case)')

    ax.axhline(y=RANDOM_BASELINE, color='black', linestyle='--',
               linewidth=1.2, label=f'Random ({RANDOM_BASELINE:.1f}%)')

    ax.set_xlabel('Defense Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Privacy-Utility — Mean ± Std (3 Seeds)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(d, d) for d in summary['defense']],
                       rotation=15, ha='right')
    ax.set_ylim(0, 118)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Right panel: ratio vs random
    if has_ratio and ratio_summary is not None:
        ax2 = axes[1]
        x2  = np.arange(len(ratio_summary))
        ax2.bar(x2, ratio_summary['ratio_mean'],
                yerr=ratio_summary['ratio_std'], capsize=4,
                color=[COLORS[d] for d in ratio_summary['defense']],
                alpha=0.85)
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5,
                    label='Random (ratio = 1.0)')
        ax2.axhline(y=2.0, color='#e74c3c', linestyle=':', linewidth=1.2, alpha=0.7,
                    label='2× — moderate leak')
        for i, (_, row) in enumerate(ratio_summary.iterrows()):
            ax2.text(i, row['ratio_mean'] + row['ratio_std'] + 0.3,
                     f"{row['ratio_mean']:.1f}×",
                     ha='center', va='bottom', fontsize=8)
        ax2.set_xlabel('Defense Method', fontsize=12)
        ax2.set_ylabel('Attack / Random Baseline', fontsize=12)
        ax2.set_title('Attack Strength vs Random\n(lower = better privacy)',
                      fontsize=13, fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels([LABELS.get(d, d) for d in ratio_summary['defense']],
                            rotation=15, ha='right')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Multi-Seed Robustness Check', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('plots/seeds_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: plots/seeds_comparison.png")


# ─────────────────────────────────────────────────────────
# Plot 7 — Per-round attack accuracy over training
# ─────────────────────────────────────────────────────────

def plot_per_round():
    path = 'results/per_round.csv'
    if not os.path.exists(path):
        print(f"  Skipping per-round plot — {path} not found")
        return

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Activity accuracy over rounds ──
    ax1 = axes[0]
    for defense in DEFENSES:
        sub = df[df['defense'] == defense].sort_values('round')
        if sub.empty:
            continue
        ax1.plot(sub['round'], sub['activity_acc'],
                 marker='o', color=COLORS[defense],
                 label=LABELS.get(defense, defense), linewidth=2)

    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Activity Accuracy (%)', fontsize=12)
    ax1.set_title('Activity Accuracy Over Rounds', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Right: Attack accuracy over rounds ──
    ax2 = axes[1]
    for defense in DEFENSES:
        sub = df[df['defense'] == defense].sort_values('round')
        if sub.empty:
            continue
        ax2.plot(sub['round'], sub['attack_acc'],
                 marker='o', color=COLORS[defense],
                 label=LABELS.get(defense, defense), linewidth=2)

    ax2.axhline(y=RANDOM_BASELINE, color='black', linestyle='--',
                linewidth=1.2, label=f'Random ({RANDOM_BASELINE:.1f}%)')

    # Mark the warmup boundary for subject_aware_delta (default = 5 rounds)
    warmup_end = 5
    ax2.axvline(x=warmup_end, color='#9b59b6', linestyle=':', linewidth=1.2, alpha=0.7)
    ax2.text(warmup_end + 0.15, ax2.get_ylim()[1] * 0.97 if ax2.get_ylim()[1] > 0 else 95,
             'SAD warmup ends →', fontsize=7.5, color='#9b59b6',
             va='top', style='italic')

    ax1.axvline(x=warmup_end, color='#9b59b6', linestyle=':', linewidth=1.2, alpha=0.5)

    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Attack Accuracy (%)', fontsize=12)
    ax2.set_title('Attack Accuracy Over Rounds\n'
                  '(does attacker improve as training progresses?)',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle('Per-Round Tracking  |  dotted line = Subject-Aware Delta warmup end',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/per_round_tracking.png', dpi=150)
    plt.close()
    print("  Saved: plots/per_round_tracking.png")


# ─────────────────────────────────────────────────────────
# Run all plots
# ─────────────────────────────────────────────────────────




# ═════════════════════════════════════════════════════════════
# ENSEMBLE PLOTS  (new in FL_Project3_Ensemble)
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────
# Plot E1 — K sweep: activity vs attack as K grows
# ─────────────────────────────────────────────────────────

def plot_ensemble_k_sweep():
    path = 'results/ensemble_k_sweep.csv'
    if not os.path.exists(path):
        print(f"  Skipping ensemble K sweep plot — {path} not found")
        return

    df = pd.read_csv(path).sort_values('n_groups')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax1, ax2 = axes

    # ── Left: Activity & Attack vs K ──
    ax1.plot(df['n_groups'], df['activity_acc'],
             marker='o', color='#3498db', linewidth=2.5, label='Activity Accuracy')
    ax1.plot(df['n_groups'], df['attack_acc'],
             marker='s', color='#e74c3c', linewidth=2.5, label='Attack Accuracy (worst-case)')
    ax1.axhline(y=RANDOM_BASELINE, color='black', linestyle='--',
                linewidth=1.2, label=f'Random ({RANDOM_BASELINE:.1f}%)')

    for _, row in df.iterrows():
        ax1.annotate(f"{row['activity_acc']:.1f}%",
                     (row['n_groups'], row['activity_acc']),
                     textcoords='offset points', xytext=(0, 8),
                     ha='center', fontsize=8, color='#3498db')
        ax1.annotate(f"{row['attack_acc']:.1f}%",
                     (row['n_groups'], row['attack_acc']),
                     textcoords='offset points', xytext=(0, -14),
                     ha='center', fontsize=8, color='#e74c3c')

    ax1.set_xlabel('Number of Groups K', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Effect of K on Privacy-Utility\n(defense=none)', fontsize=12, fontweight='bold')
    ax1.set_xticks(df['n_groups'])
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.annotate('K=1 = standard FedAvg', xy=(1, df[df['n_groups']==1]['activity_acc'].values[0]),
                 xytext=(1.3, 60), arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=8, color='gray')

    # ── Right: Privacy gain vs random & ensemble gain vs K ──
    ax2_twin = ax2.twinx()

    ax2.bar(df['n_groups'] - 0.15, df['privacy_gain_vs_random'],
            width=0.3, color='#e74c3c', alpha=0.7, label='Attack / Random (left)')
    ax2_twin.bar(df['n_groups'] + 0.15, df['ensemble_gain'],
                 width=0.3, color='#2ecc71', alpha=0.7, label='Ensemble gain pp (right)')

    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
    ax2.set_xlabel('Number of Groups K', fontsize=12)
    ax2.set_ylabel('Attack / Random Baseline', fontsize=12, color='#e74c3c')
    ax2_twin.set_ylabel('Ensemble Gain (pp)', fontsize=12, color='#2ecc71')
    ax2.set_title('Privacy Gain & Ensemble Gain vs K', fontsize=12, fontweight='bold')
    ax2.set_xticks(df['n_groups'])

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle('Ensemble FL — K Sweep (No Defense)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/ensemble_k_sweep.png', dpi=150)
    plt.close()
    print("  Saved: plots/ensemble_k_sweep.png")


# ─────────────────────────────────────────────────────────
# Plot E2 — Defense sweep at fixed K: ensemble vs baseline
# ─────────────────────────────────────────────────────────

def plot_ensemble_defense_sweep():
    ens_path = 'results/ensemble_defense_sweep.csv'
    bas_path = 'results/baseline.csv'

    if not os.path.exists(ens_path):
        print(f"  Skipping ensemble defense sweep plot — {ens_path} not found")
        return

    df_ens = pd.read_csv(ens_path)
    has_baseline = os.path.exists(bas_path)
    df_bas = pd.read_csv(bas_path) if has_baseline else None

    defenses = [d for d in DEFENSES if d in df_ens['defense'].values]
    x        = np.arange(len(defenses))
    width    = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, metric in enumerate(['activity_acc', 'attack_acc']):
        ax = axes[ax_idx]

        ens_vals = [
            float(df_ens[df_ens['defense'] == d][metric].iloc[0])
            if len(df_ens[df_ens['defense'] == d]) > 0 else 0
            for d in defenses
        ]

        bars_ens = ax.bar(x, ens_vals, width,
                          label='Ensemble FL',
                          color=[COLORS[d] for d in defenses],
                          alpha=0.85)

        if has_baseline and metric in df_bas.columns:
            bas_vals = [
                float(df_bas[df_bas['defense'] == d][metric].iloc[0])
                if len(df_bas[df_bas['defense'] == d]) > 0 else 0
                for d in defenses
            ]
            bars_bas = ax.bar(x + width, bas_vals, width,
                              label='Standard FedAvg',
                              color=[COLORS[d] for d in defenses],
                              alpha=0.4, hatch='//')
            _bar_labels(ax, bars_bas, pad=0.3, fontsize=7)

        _bar_labels(ax, bars_ens, pad=0.3, fontsize=7)

        if metric == 'attack_acc':
            ax.axhline(y=RANDOM_BASELINE, color='black', linestyle='--',
                       linewidth=1.2, label=f'Random ({RANDOM_BASELINE:.1f}%)')

        title_metric = 'Activity Accuracy' if metric == 'activity_acc' else 'Attack Accuracy (worst-case)'
        ax.set_title(f'{title_metric}\nEnsemble FL vs Standard FedAvg',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Defense Method', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([LABELS.get(d, d) for d in defenses], rotation=15, ha='right')
        ax.set_ylim(0, 118)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    k_used = int(df_ens['n_groups'].iloc[0]) if len(df_ens) > 0 else '?'
    plt.suptitle(f'Ensemble FL Defense Sweep  |  K={k_used}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/ensemble_defense_sweep.png', dpi=150)
    plt.close()
    print("  Saved: plots/ensemble_defense_sweep.png")


# ─────────────────────────────────────────────────────────
# Plot E3 — Combination: subject_aware_delta × K
# ─────────────────────────────────────────────────────────

def plot_ensemble_combination():
    path = 'results/ensemble_combination.csv'
    if not os.path.exists(path):
        print(f"  Skipping ensemble combination plot — {path} not found")
        return

    df = pd.read_csv(path).sort_values('n_groups')

    fig, ax = plt.subplots(figsize=(9, 6))

    color_act = '#9b59b6'   # purple — matches COLORS['subject_aware_delta']
    color_atk = '#e74c3c'

    ax2 = ax.twinx()

    ax.plot(df['n_groups'], df['activity_acc'],
            marker='o', color=color_act, linewidth=2.5, label='Activity Accuracy')
    ax2.plot(df['n_groups'], df['attack_acc'],
             marker='s', color=color_atk, linewidth=2.5, linestyle='--',
             label='Attack Accuracy (worst-case)')
    ax2.axhline(y=RANDOM_BASELINE, color='black', linestyle=':', linewidth=1.2,
                label=f'Random ({RANDOM_BASELINE:.1f}%)')

    for _, row in df.iterrows():
        ax.annotate(f"{row['activity_acc']:.1f}%",
                    (row['n_groups'], row['activity_acc']),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=9, color=color_act, fontweight='bold')
        ax2.annotate(f"{row['attack_acc']:.1f}%",
                     (row['n_groups'], row['attack_acc']),
                     textcoords='offset points', xytext=(0, -16),
                     ha='center', fontsize=9, color=color_atk)

    ax.set_xlabel('Number of Groups K', fontsize=12)
    ax.set_ylabel('Activity Accuracy (%)', fontsize=12, color=color_act)
    ax2.set_ylabel('Attack Accuracy (%)', fontsize=12, color=color_atk)
    ax.set_title('Subject-Aware Delta + Ensemble FL\n'
                 'Best defense × best grouping combination',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(df['n_groups'])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')
    ax.grid(alpha=0.3)

    # Shade the best operating point (highest activity, lowest attack)
    if len(df) > 0:
        best_idx = (df['activity_acc'] / (df['attack_acc'] + 1)).idxmax()
        best_k   = df.loc[best_idx, 'n_groups']
        ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(best_k + 0.05, ax.get_ylim()[1] * 0.95,
                f'Best K={best_k}', fontsize=9, color='darkgreen')

    plt.tight_layout()
    plt.savefig('plots/ensemble_combination.png', dpi=150)
    plt.close()
    print("  Saved: plots/ensemble_combination.png")


# ─────────────────────────────────────────────────────────
# Run all ensemble plots
# ─────────────────────────────────────────────────────────

def run_ensemble_plots():
    print(f"\n{'='*50}")
    print(f"  Generating ensemble plots from results/")
    print(f"{'='*50}\n")
    plot_ensemble_k_sweep()
    plot_ensemble_defense_sweep()
    plot_ensemble_combination()


if __name__ == "__main__":
    # When called directly, run ALL plots (baseline + ensemble)
    print(f"\n{'='*50}")
    print(f"  Generating ALL plots from results/")
    print(f"{'='*50}\n")

    plot_baseline()
    plot_baseline_per_clf()
    plot_ablation()
    plot_privacy_utility_scatter()
    plot_privacy_gain_vs_random()
    plot_seeds()
    plot_per_round()

    run_ensemble_plots()

    print(f"\n{'='*50}")
    print(f"  All plots saved to plots/ folder")
    print(f"{'='*50}\n")
