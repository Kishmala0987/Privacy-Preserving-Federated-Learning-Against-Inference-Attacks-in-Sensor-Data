"""
dp.py — Delta-level Differential Privacy
=========================================
Correct version: noise is applied to the WEIGHT DELTA
not to gradients during backprop.

Why delta and not w_after?
  delta = w_after - w_before
  This is exactly what the server observes per round
  Attacker also sees this exact object
  Defense and attack now operate on the same thing

Two steps applied to delta:

Step 1 — Delta Clipping:
  Clip the L2 norm of the entire delta vector
  delta = delta / max(1, ||delta|| / C)

Step 2 — Gaussian Noise:
  Add calibrated noise to clipped delta
  delta = delta + N(0, (noise_multiplier * C)^2)

Also includes subject-aware delta defense:
  Find which DIMENSIONS of the delta carry subject identity
  Add more noise to those dimensions specifically
"""

import torch
import numpy as np


def compute_delta(w_after, w_before):
    """
    Compute weight delta = w_after - w_before.

    Parameters
    ----------
    w_after  : state_dict after local training
    w_before : state_dict before local training (global model weights)

    Returns
    -------
    delta : state_dict of differences
    """
    delta = {}
    for key in w_after:
        delta[key] = w_after[key].float() - w_before[key].float()
    return delta


def clip_delta(delta, clip_threshold):
    """
    Clip the L2 norm of the delta so it does not exceed clip_threshold.

    Why clip?
      Controls sensitivity — limits how much one client's
      update can influence the aggregated result.
      Also makes noise scale meaningful relative to delta size.

    Parameters
    ----------
    delta           : state_dict of weight differences
    clip_threshold  : maximum allowed L2 norm (C)

    Returns
    -------
    clipped delta
    """
    # Compute total L2 norm across all layers
    total_norm = 0.0
    for key in delta:
        total_norm += delta[key].pow(2).sum().item()
    total_norm = total_norm ** 0.5

    # Scale down if norm exceeds threshold
    scale = max(1.0, total_norm / clip_threshold)
    clipped_delta = {}
    for key in delta:
        clipped_delta[key] = delta[key] / scale

    return clipped_delta


def add_noise_to_delta(delta, clip_threshold, noise_multiplier):
    """
    Add Gaussian noise to clipped delta.

    Noise scale = noise_multiplier * clip_threshold
    Higher noise_multiplier = more privacy, less accuracy.

    Parameters
    ----------
    delta            : clipped delta state_dict
    clip_threshold   : C used in clipping
    noise_multiplier : controls privacy-utility tradeoff

    Returns
    -------
    noisy delta
    """
    sigma = noise_multiplier * clip_threshold
    noisy_delta = {}
    for key in delta:
        noise = torch.randn_like(delta[key]) * sigma
        noisy_delta[key] = delta[key] + noise
    return noisy_delta


def apply_dp_to_delta(delta, clip_threshold, noise_multiplier):
    """
    Full DP pipeline on delta: clip then add noise.

    Parameters
    ----------
    delta            : raw weight delta
    clip_threshold   : C
    noise_multiplier : sigma multiplier

    Returns
    -------
    defended delta (clipped + noisy)
    """
    clipped = clip_delta(delta, clip_threshold)
    noisy   = add_noise_to_delta(clipped, clip_threshold, noise_multiplier)
    return noisy


# ─────────────────────────────────────────
# Subject-aware delta defense (novel)
# ─────────────────────────────────────────

def find_sensitive_delta_dimensions(deltas, subject_ids, top_k=50):
    """
    Find which DIMENSIONS of the delta vary most between subjects.

    This is the delta-level analog of subject-aware feature noise.
    Instead of finding sensitive input features,
    we find sensitive delta dimensions.

    How:
      Collect deltas from multiple clients with known subject IDs
      For each delta dimension, compute variance across subject means
      High variance = this dimension differs per subject
                    = carries subject identity in the update
                    = needs more noise

    Parameters
    ----------
    deltas      : list of flattened delta vectors (numpy arrays)
    subject_ids : list of subject IDs matching each delta
    top_k       : how many dimensions to mark as sensitive

    Returns
    -------
    sensitive_dims : indices of top_k most subject-sensitive dimensions
    """
    deltas      = np.array(deltas)       # shape: (n_samples, n_dims)
    subject_ids = np.array(subject_ids)
    unique_subjects = np.unique(subject_ids)

    # Mean delta per subject
    subject_means = np.array([
        deltas[subject_ids == s].mean(axis=0)
        for s in unique_subjects
        if (subject_ids == s).sum() > 0
    ])

    # Variance across subject means per dimension
    between_subject_var = subject_means.var(axis=0)

    # Top-k most varying dimensions
    sensitive_dims = np.argsort(between_subject_var)[::-1][:top_k].copy()
    return sensitive_dims


def apply_subject_aware_delta_defense(delta, sensitive_dims,
                                       base_noise=0.01,
                                       sensitive_multiplier=10.0):
    """
    Add more noise to subject-sensitive dimensions of the delta.

    Novel contribution:
      Normal DP adds equal noise everywhere in delta
      Subject-aware adds targeted noise to dimensions that
      most strongly identify the subject

    Parameters
    ----------
    delta                : raw delta state_dict
    sensitive_dims       : indices of sensitive dimensions in flat delta
    base_noise           : small noise added to all dimensions
    sensitive_multiplier : extra noise multiplier for sensitive dims

    Returns
    -------
    defended delta
    """
    # Flatten entire delta into one vector
    keys   = list(delta.keys())
    shapes = [delta[k].shape for k in keys]
    flat   = torch.cat([delta[k].flatten() for k in keys])

    # Add base noise to all dimensions
    noise = torch.randn_like(flat) * base_noise

    # Add extra noise to sensitive dimensions
    noise[sensitive_dims] *= sensitive_multiplier

    defended_flat = flat + noise

    # Reshape back to original layer shapes
    defended_delta = {}
    idx = 0
    for key, shape in zip(keys, shapes):
        n = 1
        for s in shape:
            n *= s
        defended_delta[key] = defended_flat[idx:idx+n].reshape(shape)
        idx += n

    return defended_delta


def flatten_delta(delta):
    """Flatten a delta state_dict into one numpy vector."""
    tensors = []
    for key in delta:
        tensors.append(delta[key].cpu().flatten().numpy())
    return np.concatenate(tensors)