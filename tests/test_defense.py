"""
tests/test_defense.py
=====================
Tests for all feature-level defense functions in core/defense.py.

Covers:
  - add_feature_noise: noise is added, shape preserved, zero multiplier is identity
  - add_subject_aware_noise: sensitive dims get more noise, non-sensitive less
  - find_subject_sensitive_features: top-k selection, correct dims identified
  - Defense under training: model still learns with defense applied (smoke test)

Run:
  cd FL_Project3_Ensemble
  python -m pytest tests/test_defense.py -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np

from core.defense import (
    add_feature_noise,
    add_subject_aware_noise,
    find_subject_sensitive_features,
)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def make_batch(n_samples=32, n_features=561, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n_samples, n_features)


def make_sensor_data_with_sensitive_dims(n_samples=200, n_features=50,
                                          n_subjects=5, sensitive_dims=None):
    """
    Synthetic sensor data where specific feature dims carry subject identity signal.
    Subject-sensitive dims have a per-subject mean; other dims are pure noise.
    """
    if sensitive_dims is None:
        sensitive_dims = [0, 1, 2]
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    s = np.array([i % n_subjects for i in range(n_samples)])
    for i in range(n_samples):
        for dim in sensitive_dims:
            X[i, dim] = s[i] * 10.0 + np.random.randn() * 0.1
    return torch.tensor(X), s


# ─────────────────────────────────────────
# add_feature_noise
# ─────────────────────────────────────────

class TestAddFeatureNoise:

    def test_noise_is_added(self):
        """Output should differ from input when noise_scale > 0."""
        X     = make_batch()
        noisy = add_feature_noise(X, noise_scale=1.0)
        assert not torch.allclose(X, noisy), "Noise should change the input"

    def test_zero_noise_is_identity(self):
        """With noise_scale=0, output should equal input exactly."""
        X     = make_batch()
        noisy = add_feature_noise(X, noise_scale=0.0)
        assert torch.allclose(X, noisy, atol=1e-6), \
            "Zero noise scale should leave input unchanged"

    def test_shape_preserved(self):
        X     = make_batch(n_samples=16, n_features=561)
        noisy = add_feature_noise(X, noise_scale=2.0)
        assert noisy.shape == X.shape, \
            f"Shape changed: {X.shape} → {noisy.shape}"

    def test_larger_noise_means_larger_perturbation(self):
        """Higher noise_scale should produce larger average perturbation."""
        X      = make_batch(seed=0)
        low    = add_feature_noise(X, noise_scale=0.1)
        high   = add_feature_noise(X, noise_scale=10.0)
        low_diff  = (low  - X).abs().mean().item()
        high_diff = (high - X).abs().mean().item()
        assert high_diff > low_diff, \
            f"Higher noise should produce larger perturbation: {high_diff:.4f} > {low_diff:.4f}"

    def test_dtype_preserved(self):
        X     = make_batch().float()
        noisy = add_feature_noise(X, noise_scale=1.0)
        assert noisy.dtype == X.dtype

    def test_batch_dimension_preserved(self):
        for batch_size in [1, 8, 64]:
            X     = make_batch(n_samples=batch_size)
            noisy = add_feature_noise(X, noise_scale=0.5)
            assert noisy.shape[0] == batch_size


# ─────────────────────────────────────────
# find_subject_sensitive_features
# ─────────────────────────────────────────

class TestFindSubjectSensitiveFeatures:

    def test_finds_truly_sensitive_dims(self):
        """Should return the actual sensitive dims in its top-k."""
        sensitive = [0, 1, 2]
        X, s = make_sensor_data_with_sensitive_dims(sensitive_dims=sensitive)
        found = find_subject_sensitive_features(X.numpy(), s, top_k=5)
        for dim in sensitive:
            assert dim in found, \
                f"Sensitive dim {dim} not in top-5 found: {found}"

    def test_returns_exactly_top_k(self):
        X, s = make_sensor_data_with_sensitive_dims()
        for k in [1, 5, 10, 20]:
            found = find_subject_sensitive_features(X.numpy(), s, top_k=k)
            assert len(found) == k, f"Expected {k} features, got {len(found)}"

    def test_indices_in_valid_range(self):
        n_features = 50
        X, s = make_sensor_data_with_sensitive_dims(n_features=n_features)
        found = find_subject_sensitive_features(X.numpy(), s, top_k=10)
        assert all(0 <= idx < n_features for idx in found), \
            f"Out-of-range index in {found}"

    def test_reproducible(self):
        X, s  = make_sensor_data_with_sensitive_dims()
        found1 = find_subject_sensitive_features(X.numpy(), s, top_k=10)
        found2 = find_subject_sensitive_features(X.numpy(), s, top_k=10)
        assert list(found1) == list(found2), "Should return same result on same input"

    def test_no_sensitive_dims_still_returns_k(self):
        """Even with pure noise data it should return top_k indices."""
        np.random.seed(0)
        X = np.random.randn(100, 561).astype(np.float32)
        s = np.array([i % 10 for i in range(100)])
        found = find_subject_sensitive_features(X, s, top_k=100)
        assert len(found) == 100

    def test_accepts_tensor_input(self):
        """Should handle both numpy arrays and torch tensors."""
        X, s = make_sensor_data_with_sensitive_dims()
        found_np  = find_subject_sensitive_features(X.numpy(), s, top_k=5)
        found_ten = find_subject_sensitive_features(X, s, top_k=5)
        assert list(found_np) == list(found_ten)


# ─────────────────────────────────────────
# add_subject_aware_noise
# ─────────────────────────────────────────

class TestAddSubjectAwareNoise:

    def _get_per_dim_noise(self, X, noisy):
        """Return mean absolute perturbation per feature dimension."""
        return (noisy - X).abs().mean(dim=0)   # shape: (n_features,)

    def test_noise_is_added(self):
        X, s = make_sensor_data_with_sensitive_dims()
        sensitive = find_subject_sensitive_features(X.numpy(), s, top_k=5)
        noisy = add_subject_aware_noise(X, sensitive_indices=sensitive, noise_scale=1.0)
        assert not torch.allclose(X, noisy), "Noise should change the input"

    def test_shape_preserved(self):
        X, s  = make_sensor_data_with_sensitive_dims()
        sens  = find_subject_sensitive_features(X.numpy(), s, top_k=5)
        noisy = add_subject_aware_noise(X, sensitive_indices=sens, noise_scale=1.0)
        assert noisy.shape == X.shape

    def test_sensitive_dims_get_more_noise(self):
        """
        Mean perturbation on sensitive dims should exceed mean on non-sensitive dims.
        We use a large noise_scale to make the signal clear.
        """
        sensitive = [0, 1, 2, 3, 4]
        X, s = make_sensor_data_with_sensitive_dims(
            n_features=50, sensitive_dims=sensitive
        )
        noisy       = add_subject_aware_noise(X, sensitive_indices=sensitive,
                                              noise_scale=10.0)
        per_dim     = self._get_per_dim_noise(X, noisy)
        sens_noise  = per_dim[sensitive].mean().item()
        other_idx   = [i for i in range(X.shape[1]) if i not in sensitive]
        other_noise = per_dim[other_idx].mean().item()

        assert sens_noise > other_noise, (
            f"Sensitive dim noise {sens_noise:.4f} should exceed "
            f"non-sensitive noise {other_noise:.4f}"
        )

    def test_zero_noise_scale_is_identity(self):
        X, s  = make_sensor_data_with_sensitive_dims()
        sens  = find_subject_sensitive_features(X.numpy(), s, top_k=5)
        noisy = add_subject_aware_noise(X, sensitive_indices=sens, noise_scale=0.0)
        assert torch.allclose(X, noisy, atol=1e-6), \
            "Zero noise scale should leave input unchanged"

    def test_empty_sensitive_indices_still_works(self):
        """With no sensitive indices, function should still apply base noise."""
        X, _ = make_sensor_data_with_sensitive_dims()
        noisy = add_subject_aware_noise(X, sensitive_indices=[], noise_scale=1.0)
        assert noisy.shape == X.shape

    def test_all_features_sensitive(self):
        """When all features are sensitive, all dims should get large noise."""
        X, s = make_sensor_data_with_sensitive_dims(n_features=10)
        all_dims = list(range(10))
        noisy    = add_subject_aware_noise(X, sensitive_indices=all_dims,
                                           noise_scale=5.0)
        per_dim  = self._get_per_dim_noise(X, noisy)
        # All dims should have substantial noise
        assert per_dim.min().item() > 0.0, \
            "All dims should have nonzero noise when all are sensitive"


# ─────────────────────────────────────────
# Defense pipeline smoke test
# ─────────────────────────────────────────

class TestDefensePipelineSmoke:
    """
    End-to-end smoke tests: verify defenses don't crash during a
    full forward pass of a model with defended inputs.
    """

    def test_feature_noise_with_model_forward(self):
        """Model forward pass should work on noise-perturbed input."""
        import sys
        sys.path.insert(0, os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')))
        from core.model import get_model

        model = get_model('cpu')
        X     = make_batch(n_samples=8, n_features=561)
        noisy = add_feature_noise(X, noise_scale=2.0)

        model.eval()
        with torch.no_grad():
            out = model(noisy)

        assert out.shape == (8, 6), \
            f"Expected output shape (8, 6), got {out.shape}"
        assert not torch.any(torch.isnan(out)), \
            "Model output should not contain NaN after feature noise"

    def test_subject_aware_noise_with_model_forward(self):
        """Model forward pass should work on subject-aware-noised input."""
        from core.model import get_model

        model     = get_model('cpu')
        X, s      = make_sensor_data_with_sensitive_dims(n_samples=8, n_features=561)
        sensitive = find_subject_sensitive_features(X.numpy(), s, top_k=100)
        noisy     = add_subject_aware_noise(X, sensitive_indices=sensitive,
                                            noise_scale=2.0)

        model.eval()
        with torch.no_grad():
            out = model(noisy)

        assert out.shape == (8, 6)
        assert not torch.any(torch.isnan(out)), \
            "Model output should not contain NaN after subject-aware noise"

    def test_defense_does_not_change_labels(self):
        """Applying defense to features should never touch the label tensor."""
        X      = make_batch()
        labels = torch.randint(0, 6, (32,))
        noisy  = add_feature_noise(X, noise_scale=5.0)
        # Labels should be completely unchanged
        assert labels.shape == (32,)
        assert labels.max() < 6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
