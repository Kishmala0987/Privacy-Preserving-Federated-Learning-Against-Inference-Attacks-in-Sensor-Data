"""
client.py — Delta-based defenses
==================================
Correct version:
  1. Train locally with CLEAN gradients
  2. Compute delta = w_after - w_before
  3. Apply defense to delta
  4. Send defended delta to server

Attacker collects the defended delta — same object defense protects.
Defense and attack now operate on the same thing.

defense_type:
  'none'                → send raw delta
  'feature'             → noise on input features during training
  'subject_aware'       → noise on subject-sensitive input features
  'dp'                  → clip + noise on delta
  'subject_aware_delta' → targeted noise on sensitive delta dimensions

Warmup phase note (subject_aware_delta only):
  During the first warmup_rounds, sensitive_delta_dims is not yet known.
  A light uniform noise (noise_multiplier=0.1) is applied as a fallback.
  These warmup deltas are slightly weaker privacy-wise and are collected
  by the attacker. This is a known limitation: the first N rounds carry
  residual subject identity leakage even under this defense.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from core.defense import add_feature_noise, add_subject_aware_noise
from core.dp import (
    compute_delta,
    apply_dp_to_delta,
    apply_subject_aware_delta_defense,
)


class FederatedClient:
    def __init__(self, client_id, subject_id, dataloader, device="cpu"):
        self.client_id             = client_id
        self.subject_id            = subject_id
        self.dataloader            = dataloader
        self.device                = device
        self.sensitive_indices     = None   # for feature-level subject_aware
        self.sensitive_delta_dims  = None   # for delta-level subject_aware

    def local_train(self, global_model, local_epochs=3, lr=0.01,
                    defense_type='none', noise_scale=0.1,
                    clip_threshold=1.0, noise_multiplier=1.0):
        """
        Train locally, compute delta, apply defense to delta, return delta.

        Parameters
        ----------
        global_model      : current global model (w_before)
        local_epochs      : passes over local data
        lr                : learning rate
        defense_type      : which defense to apply
        noise_scale       : for feature/subject_aware defenses
        clip_threshold    : for dp defense
        noise_multiplier  : for dp defense

        Returns
        -------
        defended_delta  : state_dict of defended weight differences
        num_samples     : for weighted averaging
        avg_loss        : for logging
        """

        # Save w_before — needed to compute delta at the end
        w_before = {
            key: param.data.clone()
            for key, param in global_model.named_parameters()
        }

        # Train locally with CLEAN gradients
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)

        num_samples = len(self.dataloader.dataset)
        total_loss  = 0.0
        total_steps = 0

        for epoch in range(local_epochs):
            for X_batch, y_batch in self.dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Feature-level defenses applied to input data
                if defense_type == 'feature':
                    X_batch = add_feature_noise(X_batch, noise_scale)
                elif defense_type == 'subject_aware' and self.sensitive_indices is not None:
                    X_batch = add_subject_aware_noise(
                        X_batch, self.sensitive_indices,
                        noise_scale, sensitive_multiplier=5.0
                    )

                # Clean training — no gradient noise
                optimizer.zero_grad()
                outputs = local_model(X_batch)
                loss    = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss  += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)

        # Get w_after
        w_after = local_model.state_dict()

        # Compute delta = w_after - w_before
        delta = compute_delta(w_after, w_before)

        # Apply delta-level defenses
        if defense_type == 'dp':
            # Clip delta norm then add Gaussian noise
            delta = apply_dp_to_delta(delta, clip_threshold, noise_multiplier)

        elif defense_type == 'subject_aware_delta':
            # Novel: targeted noise on subject-sensitive delta dimensions.
            # During warmup (sensitive_delta_dims is None), apply a small
            # uniform noise as a fallback so warmup deltas are not sent raw.
            # noise_scale is used as sensitive_multiplier so the ablation
            # sweep can vary this parameter meaningfully.
            if self.sensitive_delta_dims is not None:
                delta = apply_subject_aware_delta_defense(
                    delta,
                    sensitive_dims       = self.sensitive_delta_dims,
                    base_noise           = 0.01,
                    # noise_scale is used directly as sensitive_multiplier.
                    # Meaningful range is 1–50; below 1.0 adds less noise
                    # than base_noise which is not useful.
                    sensitive_multiplier = max(0.1, noise_scale),
                )
            else:
                # Warmup phase: light uniform noise on all delta dimensions
                # to avoid sending completely undefended updates while we
                # collect enough data to identify sensitive dimensions.
                from core.dp import add_noise_to_delta
                delta = add_noise_to_delta(delta,
                                           clip_threshold=1.0,
                                           noise_multiplier=0.1)

        return delta, num_samples, avg_loss