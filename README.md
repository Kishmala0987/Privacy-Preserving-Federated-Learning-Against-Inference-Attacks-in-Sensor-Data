# FL_Project3_Ensemble

Privacy-preserving Federated Learning with subject inference attack evaluation and Ensemble FL.

This project asks: *even if raw sensor data never leaves a device, can a server identify which person sent each model update?* It implements a full attack-defense evaluation pipeline and introduces a novel **subject-aware delta defense** that targets noise at the specific model-weight dimensions that carry identity signal.

---

## Installation

**Requirements:** Python 3.9+, pip

```bash
# Clone or unzip the project, then:
cd FL_Project3_Ensemble
pip install torch numpy scikit-learn matplotlib pandas pytest
```

Download the UCI-HAR dataset and place it at:
```
data/ucihar/UCI HAR Dataset/
```
The expected folder structure inside is the standard UCI-HAR layout with `train/` and `test/` subfolders.

---

## What this project does

### The threat model

30 people wear fitness trackers. Their phones train a shared model locally — raw data never leaves. The server collects only model weight updates (deltas). An honest-but-curious server runs three classifiers (Logistic Regression, Random Forest, MLP) on those deltas to re-identify which person sent which update. We report worst-case accuracy across all three classifiers.

Random guessing on 30 people = **3.3% accuracy**. Without any defense, the attack reaches ** 100%**.

### Subject-Aware Delta Defense

Standard differential privacy adds noise uniformly to all delta dimensions. This defense first identifies *which specific weight dimensions vary most between subjects* (using between-group variance analysis during a warmup phase), then adds proportionally more noise to exactly those dimensions. This preserves the activity-classification signal while masking the identity signal.

Result: **87.2% activity accuracy** with **54.8% attack accuracy** — the best privacy-utility tradeoff of all five defenses tested, at the default noise setting.

---

## Running the experiments

All commands assume you are in the project root directory.

### 1. Baseline — all five defenses

Runs standard FedAvg with each defense and saves results to `experiments/results/baseline.csv`.

```bash
python experiments/run_baseline.py
python experiments/run_baseline.py --rounds 20   # more rounds for paper results
```

**Expected output:** A table showing activity accuracy and worst-case attack accuracy for:
- `none` — no defense (attack ~97%)
- `feature` — random noise on input features
- `subject_aware` — noise on subject-sensitive input features
- `dp` — differential privacy on the delta (clip + Gaussian noise)
- `subject_aware_delta` — targeted noise on sensitive delta dimensions ★

**Expected runtime:** ~10 minutes for 15 rounds on CPU.

---

### 2. Ablation — effect of noise strength

Sweeps noise values for each defense to find the privacy-utility tradeoff curve. Saves to `experiments/results/ablation.csv`.

```bash
python experiments/run_ablation.py
python experiments/run_ablation.py --rounds 15 --defense subject_aware_delta
```

**Key finding from ablation:**
- DP sweet spot: `noise_multiplier=0.05` gives 76% activity, 11% attack
- Subject-aware delta: `sensitive_multiplier=5` gives 82% activity, 31% attack
- Both defenses show clean monotonic tradeoff curves

**Note:** The `subject_aware_delta` sweep uses `noise_scale ∈ [1, 2, 5, 10, 20, 50]` as the `sensitive_multiplier`. Values below 1.0 collapse to the same result (below the base noise floor) and are not swept.

**Expected runtime:** ~45 minutes on CPU.

---

### 3. Multi-seed robustness

Runs all defenses across 3 seeds to check variance. Saves to `experiments/results/seeds.csv`.

```bash
python experiments/run_seeds.py
```

**Known issue:** `subject_aware_delta` shows higher variance than other defenses (observed range: 17.8%–35.6% attack accuracy across 3 seeds). This is caused by random client participation in warmup rounds not always covering all 30 subjects before sensitive dimensions are computed. The `min_subjects_for_dims=10` guard in `fl_runner.py` mitigates this but does not eliminate it entirely.

**Expected runtime:** ~30 minutes on CPU.

---

### 4. Per-round tracking

Tracks how accuracy and attack success evolve round-by-round. Saves to `experiments/results/per_round.csv`.

```bash
python experiments/run_per_round.py
python experiments/run_per_round.py --rounds 30 --track_every 2
```

---

### 5. Ensemble FL experiments

Three experiments sweeping ensemble group count K.

```bash
# All three experiments (K sweep + defense sweep + combination)
python experiments/run_ensemble.py

# Individual experiments
python experiments/run_ensemble.py --exp k_sweep       # K ∈ {1,2,3,5}, no defense
python experiments/run_ensemble.py --exp defense_sweep --best_k 3
python experiments/run_ensemble.py --exp combination   # subject_aware_delta × K

# More rounds for paper results
python experiments/run_ensemble.py --rounds 20 --best_k 5
```

Ensemble *combined* with `subject_aware_delta` gives the best overall results.


---

### 6. Generate all plots

```bash
python experiments/plot_results.py
```

Plots are saved to `plots/`. Requires CSVs in `experiments/results/` from the above runs.

---

## Running the tests

```bash
# Run all 68 tests
python -m pytest tests/ -v

# Run individual test files
python -m pytest tests/test_delta.py -v      # delta computation and DP (26 tests)
python -m pytest tests/test_attacker.py -v   # subject inference attacker (20 tests)
python -m pytest tests/test_defense.py -v    # feature-level defenses (22 tests)
```

All 68 tests pass. Tests do not require the UCI-HAR dataset — they use synthetic data.

---

## Project structure

```
FL_Project3_Ensemble/
│
├── core/
│   ├── model.py              3-layer MLP: 561 → 256 → 128 → 6
│   ├── data_utils.py         UCI-HAR loading + Non-IID Dirichlet split
│   ├── client.py             Local training + all delta-level defenses
│   ├── defense.py            Feature noise and subject-aware feature noise
│   ├── dp.py                 Differential privacy + subject-aware delta defense ★
│   ├── attacker.py           3-classifier subject inference attacker
│   ├── fl_runner.py          Standard FedAvg loop
│   └── ensemble_runner.py    Ensemble FL loop (K groups, inference-time aggregation)
│
├── experiments/
│   ├── run_baseline.py       All 5 defenses under standard FedAvg
│   ├── run_ablation.py       Noise strength sweep per defense
│   ├── run_seeds.py          Multi-seed robustness check
│   ├── run_per_round.py      Per-round accuracy and attack tracking
│   ├── run_ensemble.py       3 ensemble experiments (K sweep, defense sweep, combo)
│   ├── plot_results.py       All 10 plots (7 baseline + 3 ensemble)
│   └── results/              CSV outputs from each experiment
│
├── tests/
│   ├── conftest.py           Pytest configuration
│   ├── test_delta.py         26 tests: compute_delta, clip, DP, SAD defense
│   ├── test_attacker.py      20 tests: collect, train, worst-case, report
│   └── test_defense.py       22 tests: feature noise, sensitive feature finding
│
└── README.md
```

---

## Plots generated

| File | Description |
|---|---|
| `baseline_comparison.png` | Grouped bars: activity vs attack per defense |
| `baseline_per_clf.png` | LR / RF / MLP attack accuracy side-by-side |
| `ablation_curves.png` | Noise strength vs privacy-utility curve |
| `privacy_utility_scatter.png` | Tradeoff scatter: attack vs activity |
| `privacy_gain_vs_random.png` | Attack strength as multiple of random baseline |
| `seeds_comparison.png` | Mean ± std across 3 seeds |
| `per_round_tracking.png` | Accuracy over rounds with warmup boundary |
| `ensemble_k_sweep.png` | Activity and attack as K increases |
| `ensemble_defense_sweep.png` | Ensemble vs FedAvg per defense |
| `ensemble_combination.png` | subject_aware_delta × K |

---

## Results summary

| Defense | Activity | Attack | vs Random |
|---|---|---|---|
| None | 87.8% | 97.6% | 29× |
| Feature noise | 74.5% | 94.4% | 28× |
| Subject-aware (feature) | 23.5% | 73.0% | 22× |
| DP (delta, σ=0.05) | 76.0% | 11.1% | 3.3× |
| **Subject-aware delta ★** | **87.2%** | **54.8%** | **16×** |

Subject-aware delta has the best utility (87.2% activity accuracy) while providing meaningful privacy reduction. DP achieves stronger privacy but at significant accuracy cost. The optimal operating point depends on the application's tolerance for accuracy loss.

---
