"""
defense.py
==========
Two noise-based defense mechanisms against subject inference attacks.

Defense 1 — Feature-level noise:
  Add Gaussian noise to ALL input features equally
  Simple, no knowledge of which features matter

Defense 2 — Subject-aware noise:
  Find which features differ most between subjects
  Add MORE noise to those features
  Add LESS noise to activity-related features
  Smarter — better privacy with less accuracy damage
"""

import numpy as np
import torch


# ─────────────────────────────────────────
# Defense 1: Feature-level noise
# ─────────────────────────────────────────

def add_feature_noise(X, noise_scale=0.1):
    """
    Add Gaussian noise to all features equally.

    Parameters
    ----------
    X           : input features, numpy array or torch tensor
                  shape (n_samples, 561)
    noise_scale : standard deviation of noise
                  0.01 = very small noise
                  0.1  = moderate noise
                  0.5  = heavy noise

    Returns
    -------
    X with noise added, same shape and type as input
    """
    if isinstance(X, torch.Tensor):
        noise = torch.randn_like(X) * noise_scale
        return X + noise
    else:
        noise = np.random.normal(0, noise_scale, X.shape).astype(np.float32)
        return X + noise


# ─────────────────────────────────────────
# Defense 2: Subject-aware noise
# ─────────────────────────────────────────

def find_subject_sensitive_features(X_train, s_train, top_k=50):
    """
    Find which features differ most between subjects.

    How:
      For each feature, compute variance ACROSS subject means
      High variance = this feature is different per subject
                    = carries subject identity information
                    = needs more noise

    Parameters
    ----------
    X_train  : all training features (n_samples, 561)
    s_train  : subject IDs for each sample
    top_k    : how many features to mark as subject-sensitive

    Returns
    -------
    sensitive_indices : indices of top_k most subject-sensitive features
    """
    subject_ids = np.unique(s_train)
    n_features  = X_train.shape[1]

    # Compute mean of each feature per subject
    # shape: (n_subjects, n_features)
    subject_means = np.array([
        X_train[s_train == s].mean(axis=0)
        for s in subject_ids
    ])

    # Variance across subject means per feature
    # High variance = feature differs a lot between subjects
    between_subject_variance = subject_means.var(axis=0)

    # Pick top_k features with highest between-subject variance
    sensitive_indices = np.argsort(between_subject_variance)[::-1][:top_k].copy()

    return sensitive_indices


def add_subject_aware_noise(X, sensitive_indices, noise_scale=0.1, sensitive_multiplier=3.0):
    """
    Add noise with higher intensity on subject-sensitive features.

    Sensitive features get:    noise_scale * sensitive_multiplier
    Non-sensitive features get: noise_scale (small baseline noise)

    Parameters
    ----------
    X                    : input features (n_samples, 561)
    sensitive_indices    : indices of subject-sensitive features
    noise_scale          : base noise level for all features
    sensitive_multiplier : how much extra noise on sensitive features
                           5.0 means sensitive features get 5x more noise

    Returns
    -------
    X with subject-aware noise added
    """
    if isinstance(X, torch.Tensor):
        X = X.contiguous()
        noise = torch.randn_like(X) * noise_scale
        noise = noise.clone()
        noise[..., sensitive_indices] = noise[..., sensitive_indices] * sensitive_multiplier
        return X + noise

    else:
        X = np.ascontiguousarray(X)
        noise = np.random.normal(0, noise_scale, X.shape).astype(np.float32)
        noise = noise.copy()
        noise[..., sensitive_indices] = noise[..., sensitive_indices] * sensitive_multiplier
        return X + noise

    