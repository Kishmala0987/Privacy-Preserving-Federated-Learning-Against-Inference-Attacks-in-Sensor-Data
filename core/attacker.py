"""
attacker.py
===========
Subject inference attacker — collects DELTAS from the server's perspective.

Threat model:
  Honest-but-curious server receives delta = w_after - w_before each round.
  Attacker trains a classifier on these deltas to re-identify which subject
  contributed each update.

Three classifiers are run in parallel to bracket attack strength:
  lr  — Logistic Regression      (weak attacker,  fast, linear)
  rf  — Random Forest            (medium attacker, non-linear, gives feature importances)
  mlp — Multi-Layer Perceptron   (strong attacker, non-linear, most realistic threat)

The worst-case (max across classifiers) is the number that matters for privacy claims.
If your defense holds against MLP, it's a strong result.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from collections import Counter

from core.dp import flatten_delta as _flatten_delta_base


# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────

def flatten_delta(delta):
    """
    Flatten a delta state_dict into one numpy vector.
    Wraps dp.flatten_delta and additionally replaces NaN/inf with 0,
    which can occur when very large noise values are used in the ablation sweep.
    """
    arr = _flatten_delta_base(delta)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _build_classifiers():
    """
    Return a dict of {name: classifier} for all three attack models.
    Called fresh each time so experiments don't share state.
    """
    return {
        # multi_class removed in sklearn 1.6; lbfgs handles multinomial by default
        'lr': LogisticRegression(
            max_iter    = 1000,
            random_state= 42,
            C           = 1.0,
            solver      = 'lbfgs',
        ),
        'rf': RandomForestClassifier(
            n_estimators= 200,
            max_depth   = None,       # grow full trees; delta space is wide
            random_state= 42,
            n_jobs      = -1,
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes = (256, 128),
            activation         = 'relu',
            max_iter           = 500,
            random_state       = 42,
            early_stopping     = True,
            validation_fraction= 0.1,
            n_iter_no_change   = 15,
        ),
    }


# ─────────────────────────────────────────
# Attacker
# ─────────────────────────────────────────

class SubjectInferenceAttacker:
    """
    Collects defended deltas round by round, then evaluates all three
    classifiers at the end of training (or periodically if track_per_round=True).

    Key design:
      - One shared scaler + PCA fitted once on train split.
      - RF operates on raw scaled features (trees don't need PCA).
      - LR and MLP operate on PCA-reduced features to manage dimensionality.
      - report() reuses the already-fitted objects — never re-fits.
    """

    # PCA components to keep for LR and MLP.
    # Keeps 95% of variance while reducing from ~50k dims to a manageable size.
    PCA_COMPONENTS = 200

    def __init__(self):
        self.classifiers      = _build_classifiers()
        self.scaler           = StandardScaler()
        self.pca              = PCA(n_components=self.PCA_COMPONENTS, random_state=42)

        self.is_trained       = False
        self.collected_deltas = []   # flattened delta vectors
        self.collected_labels = []   # subject IDs

        # Filled after train_and_evaluate()
        self.results          = {}   # {classifier_name: accuracy_%}
        self.rf_importances   = None # feature importances from RF

    # ── data collection ──────────────────

    def collect(self, delta, subject_id):
        """
        Save one client's defended delta with its subject label.

        Parameters
        ----------
        delta      : defended delta state_dict (what server receives)
        subject_id : which subject sent this update
        """
        flat = flatten_delta(delta)
        self.collected_deltas.append(flat)
        self.collected_labels.append(subject_id)

    # ── training and evaluation ──────────

    def train_and_evaluate(self):
        """
        Train all three classifiers on collected deltas.

        Returns
        -------
        worst_case_acc : float — max attack accuracy % across all classifiers.
                         This is the number to report in the paper.
        """
        if len(self.collected_deltas) < 10:
            print("  [attacker] Not enough data yet.")
            return 0.0

        X = np.array(self.collected_deltas)
        y = np.array(self.collected_labels)

        # Drop rows with NaN/inf (from high noise in ablation sweep)
        valid_mask = np.isfinite(X).all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 10:
            print("  [attacker] Not enough clean samples after NaN removal.")
            return 0.0

        n_unique = len(np.unique(y))
        print(f"\n  [attacker] {X.shape[0]} samples | {X.shape[1]} features | {n_unique} subjects")

        # Train/test split — stratified when possible
        min_count      = min(Counter(y).values())
        stratify_y     = y if min_count >= 2 else None
        min_test_frac  = n_unique / len(y)
        split_test_size = max(0.3, min_test_frac)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size   = split_test_size,
            random_state= 42,
            stratify    = stratify_y,
        )

        # Shared scaler (fit on train only)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        # PCA for LR and MLP (RF uses full scaled features)
        n_components = min(self.PCA_COMPONENTS, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
        self.pca     = PCA(n_components=n_components, random_state=42)
        X_train_pca  = self.pca.fit_transform(X_train_scaled)
        X_test_pca   = self.pca.transform(X_test_scaled)

        self.results = {}

        for name, clf in self.classifiers.items():
            # RF works better on the full scaled space; LR and MLP use PCA
            if name == 'rf':
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train_pca, X_test_pca

            print(f"  [attacker] Training {name.upper()}...", end=" ", flush=True)
            clf.fit(X_tr, y_train)
            acc = accuracy_score(y_test, clf.predict(X_te)) * 100
            self.results[name] = acc
            print(f"{acc:.1f}%")

        # Save RF feature importances for analysis
        self.rf_importances = self.classifiers['rf'].feature_importances_

        self.is_trained = True

        worst_case = max(self.results.values())
        return worst_case

    # ── reporting ────────────────────────

    def report(self, activity_accuracy):
        """
        Print full privacy-utility tradeoff report.
        Reuses the already-fitted scalers and classifiers — never re-fits.
        """
        if not self.is_trained:
            print("  [attacker] Not trained yet — call train_and_evaluate() first.")
            return

        random_baseline = 100.0 / len(np.unique(self.collected_labels))
        worst_case      = max(self.results.values())

        print(f"\n{'='*60}")
        print(f"  Privacy-Utility Tradeoff Report")
        print(f"{'='*60}")
        print(f"  Activity accuracy  : {activity_accuracy:.2f}%")
        print(f"  Random baseline    : {random_baseline:.2f}%  (1/{len(np.unique(self.collected_labels))} subjects)")
        print(f"")
        print(f"  {'Attacker':<10}  {'Acc':>8}  {'vs Random':>12}")
        print(f"  {'-'*36}")
        for name, acc in self.results.items():
            tag = " ← worst" if acc == worst_case else ""
            print(f"  {name.upper():<10}  {acc:>7.1f}%  {acc/random_baseline:>9.1f}x{tag}")
        print(f"")
        print(f"  Worst-case attack  : {worst_case:.2f}%")

        ratio = worst_case / random_baseline
        if worst_case > 50:
            verdict = f"SERIOUS privacy leak — {ratio:.1f}x above random. Defense insufficient."
        elif ratio > 2:
            verdict = f"MODERATE privacy leak — {ratio:.1f}x above random. Partial protection."
        else:
            verdict = f"Low privacy leak — {ratio:.1f}x above random. Defense is effective."
        print(f"  Verdict            : {verdict}")
        print(f"{'='*60}\n")

    def worst_case_acc(self):
        """Return the max attack accuracy across all classifiers (or 0 if not trained)."""
        if not self.results:
            return 0.0
        return max(self.results.values())

    def per_classifier_results(self):
        """Return {name: acc} dict — empty dict if not yet trained."""
        return dict(self.results)