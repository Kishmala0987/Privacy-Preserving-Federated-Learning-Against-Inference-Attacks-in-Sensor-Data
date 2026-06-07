"""
tests/test_attacker.py
======================
Tests for the SubjectInferenceAttacker in core/attacker.py.

Covers:
  - collect(): data accumulation
  - train_and_evaluate(): trains all three classifiers, returns worst-case
  - worst_case_acc(): returns max across classifiers
  - per_classifier_results(): returns all three values
  - report(): runs without crash
  - Distinguishable deltas: attacker should do well on clearly different deltas
  - Indistinguishable deltas: attacker should be near random on pure noise

Run:
  cd FL_Project3_Ensemble
  python -m pytest tests/test_attacker.py -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import torch

from core.attacker import SubjectInferenceAttacker, flatten_delta


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def make_delta(mean=0.0, std=0.1, size=500, seed=None):
    """Create a fake delta state dict with one layer."""
    if seed is not None:
        torch.manual_seed(seed)
    return {'layer0': torch.randn(size) * std + mean}


def feed_attacker(attacker, n_subjects=10, n_rounds=5,
                  distinguishable=True, delta_size=500):
    """
    Feed synthetic deltas to the attacker.

    distinguishable=True  → each subject has a unique mean per delta dim
                            (attacker should achieve high accuracy)
    distinguishable=False → all subjects have identical distributions
                            (attacker should be near random)
    """
    np.random.seed(0)
    for subject in range(n_subjects):
        for _ in range(n_rounds):
            if distinguishable:
                # Each subject has a unique, strong signal
                mean = subject * 5.0
                delta = {'layer0': torch.randn(delta_size) * 0.01 + mean}
            else:
                # All subjects look the same
                delta = {'layer0': torch.randn(delta_size) * 1.0}
            attacker.collect(delta, subject_id=subject)


# ─────────────────────────────────────────
# collect()
# ─────────────────────────────────────────

class TestCollect:

    def test_collects_deltas(self):
        """Collected deltas should accumulate correctly."""
        attacker = SubjectInferenceAttacker()
        for i in range(5):
            attacker.collect(make_delta(), subject_id=i)
        assert len(attacker.collected_deltas) == 5
        assert len(attacker.collected_labels) == 5

    def test_labels_match_subjects(self):
        """Labels should match the subject IDs passed to collect()."""
        attacker = SubjectInferenceAttacker()
        for subject_id in [1, 5, 10, 22, 30]:
            attacker.collect(make_delta(), subject_id=subject_id)
        assert attacker.collected_labels == [1, 5, 10, 22, 30]

    def test_flattening_happens(self):
        """Stored deltas should be flat numpy vectors, not dicts."""
        attacker = SubjectInferenceAttacker()
        attacker.collect({'layer0': torch.ones(100), 'layer1': torch.ones(50)}, subject_id=0)
        assert isinstance(attacker.collected_deltas[0], np.ndarray)
        assert attacker.collected_deltas[0].shape == (150,)

    def test_nan_replaced(self):
        """NaN/inf values in delta should be replaced with 0."""
        attacker = SubjectInferenceAttacker()
        bad_delta = {'layer0': torch.tensor([float('nan'), float('inf'), 1.0])}
        attacker.collect(bad_delta, subject_id=0)
        flat = attacker.collected_deltas[0]
        assert np.all(np.isfinite(flat)), "NaN/inf should be replaced with 0"


# ─────────────────────────────────────────
# train_and_evaluate()
# ─────────────────────────────────────────

class TestTrainAndEvaluate:

    def test_returns_float(self):
        attacker = SubjectInferenceAttacker()
        feed_attacker(attacker, n_subjects=5, n_rounds=8)
        result = attacker.train_and_evaluate()
        assert isinstance(result, float)

    def test_result_in_valid_range(self):
        attacker = SubjectInferenceAttacker()
        feed_attacker(attacker, n_subjects=5, n_rounds=8)
        result = attacker.train_and_evaluate()
        assert 0.0 <= result <= 100.0, f"Accuracy {result} out of [0, 100] range"

    def test_all_three_classifiers_trained(self):
        attacker = SubjectInferenceAttacker()
        feed_attacker(attacker, n_subjects=5, n_rounds=8)
        attacker.train_and_evaluate()
        assert 'lr'  in attacker.results
        assert 'rf'  in attacker.results
        assert 'mlp' in attacker.results

    def test_too_few_samples_returns_zero(self):
        """With fewer than 10 samples train_and_evaluate should return 0."""
        attacker = SubjectInferenceAttacker()
        attacker.collect(make_delta(), subject_id=0)
        attacker.collect(make_delta(), subject_id=1)
        result = attacker.train_and_evaluate()
        assert result == 0.0

    def test_high_accuracy_on_distinguishable_deltas(self):
        """
        When each subject has a clearly unique delta pattern,
        the attacker should achieve significantly above-random accuracy.
        Random chance = 100/10 = 10%.
        """
        attacker = SubjectInferenceAttacker()
        feed_attacker(attacker, n_subjects=10, n_rounds=10, distinguishable=True)
        acc = attacker.train_and_evaluate()
        random_chance = 100.0 / 10
        assert acc > random_chance * 2, (
            f"Expected attacker to significantly beat random ({random_chance:.1f}%), "
            f"got {acc:.1f}%"
        )

    def test_near_random_on_indistinguishable_deltas(self):
        """
        When all subjects have identical delta distributions,
        the attacker should be near random chance.
        Random chance = 100/10 = 10%.
        """
        attacker = SubjectInferenceAttacker()
        feed_attacker(attacker, n_subjects=10, n_rounds=20, distinguishable=False)
        acc = attacker.train_and_evaluate()
        random_chance = 100.0 / 10
        # Allow generous margin — classifiers might overfit slightly
        assert acc < random_chance * 4, (
            f"On noise-only deltas, attacker accuracy {acc:.1f}% is too high "
            f"(expected near random={random_chance:.1f}%)"
        )

    def test_is_trained_flag_set(self):
        attacker = SubjectInferenceAttacker()
        assert not attacker.is_trained
        feed_attacker(attacker, n_subjects=5, n_rounds=8)
        attacker.train_and_evaluate()
        assert attacker.is_trained


# ─────────────────────────────────────────
# worst_case_acc() and per_classifier_results()
# ─────────────────────────────────────────

class TestResultAccessors:

    def test_worst_case_is_max(self):
        """worst_case_acc() should return max across all three classifiers."""
        attacker = SubjectInferenceAttacker()
        feed_attacker(attacker, n_subjects=5, n_rounds=8)
        attacker.train_and_evaluate()
        worst  = attacker.worst_case_acc()
        all_acc = list(attacker.per_classifier_results().values())
        assert worst == max(all_acc), \
            f"worst_case_acc()={worst} should equal max({all_acc})"

    def test_worst_case_before_training(self):
        """worst_case_acc() before training should return 0."""
        attacker = SubjectInferenceAttacker()
        assert attacker.worst_case_acc() == 0.0

    def test_per_classifier_results_keys(self):
        attacker = SubjectInferenceAttacker()
        feed_attacker(attacker, n_subjects=5, n_rounds=8)
        attacker.train_and_evaluate()
        results = attacker.per_classifier_results()
        assert set(results.keys()) == {'lr', 'rf', 'mlp'}

    def test_per_classifier_before_training(self):
        attacker = SubjectInferenceAttacker()
        assert attacker.per_classifier_results() == {}


# ─────────────────────────────────────────
# report()
# ─────────────────────────────────────────

class TestReport:

    def test_report_runs_without_crash(self, capsys):
        attacker = SubjectInferenceAttacker()
        feed_attacker(attacker, n_subjects=5, n_rounds=8)
        attacker.train_and_evaluate()
        attacker.report(activity_accuracy=75.0)   # should not raise
        captured = capsys.readouterr()
        assert 'Privacy-Utility' in captured.out

    def test_report_before_training_prints_warning(self, capsys):
        attacker = SubjectInferenceAttacker()
        attacker.report(activity_accuracy=80.0)
        captured = capsys.readouterr()
        assert 'Not trained' in captured.out

    def test_report_contains_verdict(self, capsys):
        attacker = SubjectInferenceAttacker()
        feed_attacker(attacker, n_subjects=5, n_rounds=8, distinguishable=True)
        attacker.train_and_evaluate()
        attacker.report(activity_accuracy=85.0)
        captured = capsys.readouterr()
        # Should contain one of the three verdict strings
        assert any(v in captured.out for v in ['SERIOUS', 'MODERATE', 'Low']), \
            f"Expected verdict in output. Got:\n{captured.out}"


# ─────────────────────────────────────────
# flatten_delta (attacker-side wrapper)
# ─────────────────────────────────────────

class TestFlattenDeltaAttacker:
    """
    The attacker has its own flatten_delta wrapper that replaces NaN/inf.
    Test that it handles bad inputs gracefully.
    """

    def test_normal_delta(self):
        delta = {'a': torch.tensor([1.0, 2.0]), 'b': torch.tensor([3.0])}
        flat  = flatten_delta(delta)
        assert flat.shape == (3,)
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0], rtol=1e-5)

    def test_nan_replaced_with_zero(self):
        delta = {'a': torch.tensor([float('nan'), 1.0, float('inf')])}
        flat  = flatten_delta(delta)
        assert np.all(np.isfinite(flat)), f"Expected finite values, got {flat}"
        assert flat[0] == 0.0 and flat[2] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
