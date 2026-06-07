"""
fl_runner.py
============
Pure FL logic — no experiments, no CSV saving, no plotting.

This is the engine that all experiment files call.
Takes a config dict, runs FL, returns results dict.

Used by:
  experiments/run_baseline.py
  experiments/run_ablation.py
  experiments/run_seeds.py
"""

import copy
import random
import torch
import torch.nn as nn
import numpy as np

from core.data_utils import (
    load_ucihar,
    create_federated_clients,
    get_client_dataloader,
    get_test_dataloader,
)
from core.model import get_model
from core.client import FederatedClient
from core.attacker import SubjectInferenceAttacker
from core.defense import find_subject_sensitive_features
from core.dp import find_sensitive_delta_dimensions, flatten_delta


# ─────────────────────────────────────────
# FedAvg on deltas
# ─────────────────────────────────────────

def fedavg_delta(global_model, deltas, client_sizes):
    total     = sum(client_sizes)
    avg_delta = copy.deepcopy(deltas[0])

    for key in avg_delta:
        avg_delta[key] = torch.zeros_like(avg_delta[key], dtype=torch.float32)

    for delta, n_k in zip(deltas, client_sizes):
        fraction = n_k / total
        for key in avg_delta:
            avg_delta[key] += fraction * delta[key].float()

    global_state = global_model.state_dict()
    new_state    = {}
    for key in global_state:
        if key in avg_delta:
            new_state[key] = global_state[key].float() + avg_delta[key]
        else:
            new_state[key] = global_state[key]

    global_model.load_state_dict(new_state)
    return global_model


# ─────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────

def evaluate(model, test_loader, device):
    model.eval()
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total   += y_batch.size(0)
    return 100.0 * correct / total, total_loss / len(test_loader)


# ─────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────

def run_one_experiment(config):
    """
    Run one full FL experiment with one defense configuration.

    Parameters
    ----------
    config : dict with keys:
        seed                : random seed
        rounds              : communication rounds
        participation_rate  : fraction of clients per round
        alpha               : Dirichlet skew
        local_epochs        : local training epochs
        lr                  : learning rate
        batch_size          : local batch size
        defense_type        : 'none','feature','subject_aware','dp','subject_aware_delta'
        noise_scale         : for feature defenses
        clip_threshold      : for dp
        noise_multiplier    : for dp
        warmup_rounds       : rounds before subject_aware_delta activates
        track_per_round     : bool — track attack accuracy every N rounds
        track_every         : int  — how often to track (if track_per_round=True)

    Returns
    -------
    results : dict with:
        final_activity_acc       : float
        final_attack_acc         : float (worst-case across all classifiers)
        attack_acc_per_clf       : dict {clf_name: acc}
        privacy_gain             : float (needs baseline to compute, done externally)
        privacy_gain_vs_random   : float — attack_acc / (100/n_subjects); returned
                                   here for convenience; 1.0 = random-level privacy
        per_round_acc            : list of activity accuracy per round
        per_round_attack         : list of (round, attack_acc) if track_per_round=True

    Note on subject/client mapping:
        UCI-HAR has 30 subjects and we create one client per subject (1:1 mapping).
        This is realistic for HAR but means the attacker's task is easier than in
        deployments where multiple subjects share a device. Attack accuracies should
        be interpreted in this context.
    """

    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    defense_type = config['defense_type']

    # Load data (cached after first call)
    X_train, y_train, s_train, X_test, y_test, s_test = load_ucihar()

    # Find subject-sensitive input features
    sensitive_indices = find_subject_sensitive_features(
        X_train, s_train, top_k=100
    )

    # Create clients
    client_data = create_federated_clients(
        X_train, y_train, s_train,
        alpha=config.get('alpha', 0.5),
        seed=seed,
    )

    clients = []
    for i, cd in enumerate(client_data):
        loader = get_client_dataloader(
            cd, batch_size=config.get('batch_size', 32)
        )
        clients.append(FederatedClient(
            client_id  = i,
            subject_id = cd["subject_id"],
            dataloader = loader,
            device     = device,
        ))

    test_loader  = get_test_dataloader(X_test, y_test)
    global_model = get_model(device)
    attacker     = SubjectInferenceAttacker()

    num_total       = len(clients)
    rounds          = config.get('rounds', 15)
    warmup_rounds   = config.get('warmup_rounds', 5)
    track_per_round = config.get('track_per_round', False)
    track_every     = config.get('track_every', 5)

    per_round_acc    = []
    per_round_attack = []

    warmup_deltas       = []
    warmup_subject_ids  = []
    sensitive_delta_dims = None

    for rnd in range(1, rounds + 1):

        num_selected = max(1, int(num_total * config.get('participation_rate', 0.7)))
        selected     = random.sample(clients, num_selected)

        # Compute sensitive delta dimensions at the start of the first
        # post-warmup round, before any client runs local_train this round.
        # This guarantees every client in this round gets the sensitive dims.
        if (defense_type == 'subject_aware_delta'
                and rnd == warmup_rounds + 1
                and sensitive_delta_dims is None):
            n_unique_seen = len(set(warmup_subject_ids))
            # Need at least 10 unique subjects to get a meaningful
            # between-subject variance estimate for sensitive dims.
            # If warmup didn't cover enough subjects (can happen with
            # low participation rate), extend warmup by 1 round.
            min_subjects = config.get('min_subjects_for_dims', 10)
            if len(warmup_deltas) > 10 and n_unique_seen >= min_subjects:
                sensitive_delta_dims = find_sensitive_delta_dimensions(
                    warmup_deltas, warmup_subject_ids,
                    top_k=config.get('sensitive_top_k', 50)
                )
                for c in clients:
                    c.sensitive_delta_dims = sensitive_delta_dims
                print(f'  [SAD] Sensitive dims computed at round {rnd} '
                      f'({n_unique_seen} subjects seen in warmup)')
            else:
                print(f'  [SAD] Warmup insufficient ({n_unique_seen} subjects, '
                      f'{len(warmup_deltas)} deltas) — extending warmup 1 round')

        deltas       = []
        client_sizes = []

        for client in selected:

            # Set sensitive info on client
            client.sensitive_indices = (
                sensitive_indices
                if defense_type == 'subject_aware' else None
            )

            delta, n, _ = client.local_train(
                global_model     = global_model,
                local_epochs     = config.get('local_epochs', 3),
                lr               = config.get('lr', 0.01),
                defense_type     = defense_type,
                noise_scale      = config.get('noise_scale', 2.0),
                clip_threshold   = config.get('clip_threshold', 1.0),
                noise_multiplier = config.get('noise_multiplier', 1.0),
            )

            deltas.append(delta)
            client_sizes.append(n)
            attacker.collect(delta, client.subject_id)

            if defense_type == 'subject_aware_delta' and rnd <= warmup_rounds:
                warmup_deltas.append(flatten_delta(delta))
                warmup_subject_ids.append(client.subject_id)

        global_model = fedavg_delta(global_model, deltas, client_sizes)
        acc, _       = evaluate(global_model, test_loader, device)
        per_round_acc.append(acc)

        # Per-round attack tracking
        if track_per_round and rnd % track_every == 0:
            if len(attacker.collected_deltas) >= 10:
                attacker.train_and_evaluate()
                per_round_attack.append((rnd, attacker.worst_case_acc(), attacker.per_classifier_results()))

    # Final attack evaluation — runs all three classifiers
    attacker.train_and_evaluate()

    # Random-chance baseline for 30 subjects
    random_chance = 100.0 / len(set(c.subject_id for c in clients))

    return {
        'final_activity_acc'     : per_round_acc[-1],
        # Worst-case across all classifiers — the number to report in papers
        'final_attack_acc'       : attacker.worst_case_acc(),
        # Per-classifier breakdown for detailed analysis
        'attack_acc_per_clf'     : attacker.per_classifier_results(),
        # How many times above random is the attack? 1.0 = perfect privacy
        'privacy_gain_vs_random' : round(attacker.worst_case_acc() / random_chance, 4),
        'per_round_acc'          : per_round_acc,
        'per_round_attack'       : per_round_attack,
    }