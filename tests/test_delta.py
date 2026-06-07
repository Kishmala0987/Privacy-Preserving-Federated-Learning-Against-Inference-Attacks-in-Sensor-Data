"""
tests/test_delta.py
===================
Tests for delta computation and all delta-level defense functions in core/dp.py.

Covers:
  - compute_delta: correctness, zero delta, shape preservation
  - clip_delta: norm clipping, already-small delta not clipped
  - add_noise_to_delta: noise is added, shape preserved
  - apply_dp_to_delta: combined clip+noise pipeline
  - find_sensitive_delta_dimensions: top-k selection, reproducibility
  - apply_subject_aware_delta_defense: sensitive dims get more noise
  - flatten_delta: correct concatenation and shape

Run:
  cd FL_Project3_Ensemble
  python -m pytest tests/test_delta.py -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np

from core.dp import (
    compute_delta,
    clip_delta,
    add_noise_to_delta,
    apply_dp_to_delta,
    find_sensitive_delta_dimensions,
    apply_subject_aware_delta_defense,
    flatten_delta,
)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def make_state_dict(shapes, fill=0.0):
    """Create a fake state dict with given shapes, filled with a constant."""
    return {f'layer{i}': torch.full(s, fill) for i, s in enumerate(shapes)}


def make_random_state_dict(shapes, seed=0):
    torch.manual_seed(seed)
    return {f'layer{i}': torch.randn(s) for i, s in enumerate(shapes)}


SHAPES = [(256, 561), (256,), (128, 256), (128,), (6, 128), (6,)]


# ─────────────────────────────────────────
# compute_delta
# ─────────────────────────────────────────

class TestComputeDelta:

    def test_correct_difference(self):
        """delta[k] should equal w_after[k] - w_before[k] elementwise."""
        w_before = make_state_dict([(3, 4)], fill=1.0)
        w_after  = make_state_dict([(3, 4)], fill=3.0)
        delta    = compute_delta(w_after, w_before)
        expected = torch.full((3, 4), 2.0)
        assert torch.allclose(delta['layer0'], expected), \
            f"Expected delta=2.0, got {delta['layer0'].mean()}"

    def test_zero_delta_when_equal(self):
        """If w_after == w_before the delta should be all zeros."""
        w = make_random_state_dict(SHAPES, seed=1)
        delta = compute_delta(w, w)
        for key, tensor in delta.items():
            assert torch.allclose(tensor, torch.zeros_like(tensor)), \
                f"Expected zero delta for key={key}"

    def test_shape_preserved(self):
        """Delta should have the same shape as the input weights."""
        w_before = make_random_state_dict(SHAPES, seed=2)
        w_after  = make_random_state_dict(SHAPES, seed=3)
        delta    = compute_delta(w_after, w_before)
        for key in w_before:
            assert delta[key].shape == w_before[key].shape, \
                f"Shape mismatch for {key}: {delta[key].shape} vs {w_before[key].shape}"

    def test_all_keys_present(self):
        """Delta should have exactly the same keys as the input dicts."""
        w_before = make_random_state_dict(SHAPES, seed=4)
        w_after  = make_random_state_dict(SHAPES, seed=5)
        delta    = compute_delta(w_after, w_before)
        assert set(delta.keys()) == set(w_before.keys())

    def test_dtype_is_float(self):
        """Delta values should be float32 regardless of input dtype."""
        w_before = {k: v.double() for k, v in make_random_state_dict([(4, 4)]).items()}
        w_after  = {k: v.double() for k, v in make_random_state_dict([(4, 4)], seed=1).items()}
        delta    = compute_delta(w_after, w_before)
        for key, tensor in delta.items():
            assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"


# ─────────────────────────────────────────
# clip_delta
# ─────────────────────────────────────────

class TestClipDelta:

    def test_large_delta_gets_clipped(self):
        """A delta with large norm should be clipped so its L2 norm <= clip_threshold."""
        delta    = make_state_dict([(100,)], fill=1.0)   # norm = 10
        clipped  = clip_delta(delta, clip_threshold=1.0)
        flat     = torch.cat([clipped[k].flatten() for k in clipped])
        norm     = flat.norm().item()
        assert norm <= 1.0 + 1e-5, f"Clipped norm {norm:.4f} exceeds threshold 1.0"

    def test_small_delta_not_clipped(self):
        """A delta already within the threshold should not be changed."""
        delta   = make_state_dict([(4,)], fill=0.1)   # norm = 0.2 < 1.0
        clipped = clip_delta(delta, clip_threshold=1.0)
        for key in delta:
            assert torch.allclose(delta[key], clipped[key], atol=1e-6), \
                "Small delta should not be modified by clipping"

    def test_clipping_is_proportional(self):
        """Clipping should scale all layers uniformly, not zero them out."""
        delta   = make_state_dict([(2, 2)], fill=2.0)
        clipped = clip_delta(delta, clip_threshold=1.0)
        # All values should be equal (same scale applied everywhere)
        vals = clipped['layer0'].unique()
        assert len(vals) == 1, "Clipping should produce uniform scaling"

    def test_shape_preserved(self):
        delta   = make_random_state_dict(SHAPES)
        clipped = clip_delta(delta, clip_threshold=0.5)
        for key in delta:
            assert clipped[key].shape == delta[key].shape


# ─────────────────────────────────────────
# add_noise_to_delta
# ─────────────────────────────────────────

class TestAddNoiseToDelta:

    def test_noise_is_added(self):
        """After adding noise the delta should differ from the original."""
        delta = make_state_dict(SHAPES, fill=1.0)
        noisy = add_noise_to_delta(delta, clip_threshold=1.0, noise_multiplier=1.0)
        any_different = any(
            not torch.allclose(delta[k], noisy[k]) for k in delta
        )
        assert any_different, "Noise should change at least some values"

    def test_zero_noise_multiplier(self):
        """With noise_multiplier=0 the output should equal the input."""
        delta = make_random_state_dict(SHAPES)
        noisy = add_noise_to_delta(delta, clip_threshold=1.0, noise_multiplier=0.0)
        for key in delta:
            assert torch.allclose(delta[key], noisy[key], atol=1e-6), \
                f"Zero noise multiplier should not change delta for key={key}"

    def test_shape_preserved(self):
        delta = make_random_state_dict(SHAPES)
        noisy = add_noise_to_delta(delta, clip_threshold=1.0, noise_multiplier=0.5)
        for key in delta:
            assert noisy[key].shape == delta[key].shape


# ─────────────────────────────────────────
# apply_dp_to_delta
# ─────────────────────────────────────────

class TestApplyDpToDelta:

    def test_norm_bounded_after_dp(self):
        """After DP, delta norm should be approximately bounded by clip_threshold."""
        # With noise_multiplier=0 (pure clipping), norm must be <= threshold
        delta   = make_state_dict(SHAPES, fill=5.0)
        dp      = apply_dp_to_delta(delta, clip_threshold=1.0, noise_multiplier=0.0)
        flat    = torch.cat([dp[k].flatten() for k in dp])
        norm    = flat.norm().item()
        assert norm <= 1.0 + 1e-4, f"DP norm {norm:.4f} should be <= 1.0 after clipping"

    def test_output_differs_from_input(self):
        """DP output should differ from raw input (clipping + noise change it)."""
        delta = make_state_dict(SHAPES, fill=5.0)
        dp    = apply_dp_to_delta(delta, clip_threshold=1.0, noise_multiplier=0.1)
        any_different = any(not torch.allclose(delta[k], dp[k]) for k in delta)
        assert any_different

    def test_shape_preserved(self):
        delta = make_random_state_dict(SHAPES)
        dp    = apply_dp_to_delta(delta, clip_threshold=1.0, noise_multiplier=0.5)
        for key in delta:
            assert dp[key].shape == delta[key].shape


# ─────────────────────────────────────────
# find_sensitive_delta_dimensions
# ─────────────────────────────────────────

class TestFindSensitiveDeltaDimensions:

    def _make_deltas_with_sensitive_dims(self, n_dims=100, n_subjects=5,
                                          n_per_subject=10, sensitive=[0, 1, 2]):
        """
        Construct synthetic deltas where specific dims clearly vary between subjects.
        Non-sensitive dims are random noise; sensitive dims have a subject-specific mean.
        """
        np.random.seed(99)
        deltas, labels = [], []
        for s in range(n_subjects):
            for _ in range(n_per_subject):
                d = np.random.randn(n_dims).astype(np.float32)
                # Sensitive dims get a strong subject-specific shift
                for dim in sensitive:
                    d[dim] = s * 10.0 + np.random.randn() * 0.1
                deltas.append(d)
                labels.append(s)
        return deltas, labels

    def test_finds_correct_sensitive_dims(self):
        """The function should rank truly sensitive dims in the top-k."""
        sensitive = [0, 1, 2]
        deltas, labels = self._make_deltas_with_sensitive_dims(
            n_dims=50, sensitive=sensitive
        )
        found = find_sensitive_delta_dimensions(deltas, labels, top_k=5)
        # All 3 truly sensitive dims should appear in the top 5
        for dim in sensitive:
            assert dim in found, \
                f"Sensitive dim {dim} not found in top-5: {found}"

    def test_returns_top_k(self):
        """Should return exactly top_k indices."""
        deltas, labels = self._make_deltas_with_sensitive_dims()
        found = find_sensitive_delta_dimensions(deltas, labels, top_k=10)
        assert len(found) == 10

    def test_reproducible(self):
        """Same input should always produce the same output."""
        deltas, labels = self._make_deltas_with_sensitive_dims()
        found1 = find_sensitive_delta_dimensions(deltas, labels, top_k=10)
        found2 = find_sensitive_delta_dimensions(deltas, labels, top_k=10)
        assert list(found1) == list(found2)

    def test_indices_in_range(self):
        """All returned indices should be valid dimension indices."""
        n_dims = 50
        deltas = [np.random.randn(n_dims).astype(np.float32) for _ in range(20)]
        labels = [i % 5 for i in range(20)]
        found  = find_sensitive_delta_dimensions(deltas, labels, top_k=10)
        assert all(0 <= idx < n_dims for idx in found), \
            f"Out-of-range index in {found}"


# ─────────────────────────────────────────
# apply_subject_aware_delta_defense
# ─────────────────────────────────────────

class TestApplySubjectAwareDeltaDefense:

    def _make_delta(self, shapes=None):
        shapes = shapes or SHAPES
        torch.manual_seed(7)
        return {f'layer{i}': torch.randn(s) for i, s in enumerate(shapes)}

    def _get_flat_noise(self, original, defended):
        """Return the absolute noise added per dimension."""
        orig_flat = torch.cat([original[k].flatten() for k in original])
        def_flat  = torch.cat([defended[k].flatten() for k in defended])
        return (def_flat - orig_flat).abs()

    def test_sensitive_dims_get_more_noise(self):
        """
        Sensitive dimensions should have higher absolute noise than non-sensitive ones.
        We test this statistically: mean noise on sensitive dims > mean noise on others.
        """
        delta         = self._make_delta()
        flat_size     = sum(v.numel() for v in delta.values())
        sensitive_dims = np.array([0, 1, 2, 3, 4])   # first 5 dims

        defended = apply_subject_aware_delta_defense(
            delta,
            sensitive_dims       = sensitive_dims,
            base_noise           = 0.001,
            sensitive_multiplier = 50.0,   # large multiplier for clear signal
        )

        noise       = self._get_flat_noise(delta, defended)
        sens_noise  = noise[sensitive_dims].mean().item()
        other_mask  = torch.ones(flat_size, dtype=torch.bool)
        other_mask[sensitive_dims] = False
        other_noise = noise[other_mask].mean().item()

        assert sens_noise > other_noise * 5, (
            f"Sensitive dims noise {sens_noise:.4f} should be >> "
            f"non-sensitive noise {other_noise:.4f}"
        )

    def test_shape_preserved(self):
        delta    = self._make_delta()
        flat_size = sum(v.numel() for v in delta.values())
        sensitive = np.arange(10)
        defended  = apply_subject_aware_delta_defense(
            delta, sensitive_dims=sensitive, base_noise=0.01, sensitive_multiplier=5.0
        )
        for key in delta:
            assert defended[key].shape == delta[key].shape

    def test_base_noise_applied_everywhere(self):
        """Even non-sensitive dims should have some noise (base_noise > 0)."""
        delta         = self._make_delta()
        flat_size     = sum(v.numel() for v in delta.values())
        sensitive_dims = np.array([0])   # only dim 0 is sensitive
        defended = apply_subject_aware_delta_defense(
            delta,
            sensitive_dims       = sensitive_dims,
            base_noise           = 1.0,   # large base noise so signal is clear
            sensitive_multiplier = 1.0,   # same multiplier — no difference expected
        )
        noise = self._get_flat_noise(delta, defended)
        # All dims should have nonzero noise due to base_noise
        assert noise.mean().item() > 0.1, \
            "Base noise should be present on all dimensions"

    def test_zero_multiplier_equals_base_noise(self):
        """sensitive_multiplier=1.0 means sensitive dims get same noise as others."""
        delta         = self._make_delta()
        sensitive_dims = np.array([0, 1, 2])
        torch.manual_seed(42)
        defended = apply_subject_aware_delta_defense(
            delta,
            sensitive_dims       = sensitive_dims,
            base_noise           = 0.1,
            sensitive_multiplier = 1.0,
        )
        noise = self._get_flat_noise(delta, defended)
        sens  = noise[sensitive_dims].mean().item()
        flat_size = sum(v.numel() for v in delta.values())
        other_mask = torch.ones(flat_size, dtype=torch.bool)
        other_mask[sensitive_dims] = False
        other = noise[other_mask].mean().item()
        # With multiplier=1, both should be similar (within 3x of each other)
        ratio = sens / (other + 1e-9)
        assert ratio < 3.0, \
            f"With multiplier=1, noise ratio sens/other={ratio:.2f} should be ~1"


# ─────────────────────────────────────────
# flatten_delta
# ─────────────────────────────────────────

class TestFlattenDelta:

    def test_correct_total_size(self):
        """Flattened vector should have length = sum of all parameter counts."""
        delta    = make_random_state_dict(SHAPES)
        expected = sum(1 for s in SHAPES for _ in range(s[0] * (s[1] if len(s) > 1 else 1)))
        flat     = flatten_delta(delta)
        total    = sum(torch.tensor(s).prod().item() for s in SHAPES)
        assert len(flat) == total, f"Expected {total} elements, got {len(flat)}"

    def test_returns_numpy(self):
        delta = make_random_state_dict([(4, 4)])
        flat  = flatten_delta(delta)
        assert isinstance(flat, np.ndarray), "flatten_delta should return numpy array"

    def test_values_match_original(self):
        """First few values of flat vector should match first layer's values."""
        delta  = make_random_state_dict([(3,)])
        flat   = flatten_delta(delta)
        original_vals = delta['layer0'].numpy()
        np.testing.assert_allclose(flat[:3], original_vals, rtol=1e-5)

    def test_single_layer(self):
        delta = {'w': torch.tensor([1.0, 2.0, 3.0])}
        flat  = flatten_delta(delta)
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
