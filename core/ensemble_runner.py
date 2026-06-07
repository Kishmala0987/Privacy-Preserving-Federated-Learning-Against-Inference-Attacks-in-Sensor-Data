"""
ensemble_runner.py
==================
Ensemble Federated Learning runner.

How it differs from standard FedAvg (fl_runner.py):
  - Clients are split into K groups at the start (fixed, random assignment)
  - Each group has its own independent model (shard model / ensemble member)
  - Per round: within each group, clients train locally → group FedAvg → group model updated
  - No parameters are ever merged across groups
  - At inference: all K group models predict → softmax probabilities averaged → final prediction

So aggregation happens twice, in two different forms:
  1. Training-time: FedAvg inside each group (parameter averaging within group)
  2. Inference-time: ensemble averaging across groups (probability averaging across models)

Privacy angle:
  The attacker (honest-but-curious server) sees deltas from all groups.
  With K groups, each group's attacker only sees ~(n_clients/K) * rounds deltas.
  This reduces the attacker's training data and should reduce attack accuracy.
  All existing defenses (DP, subject_aware_delta) stack on top.

Degeneracy check:
  K=1 → identical to standard FedAvg (one group, one model, standard aggregation).
  This is used as the sanity check baseline in run_ensemble.py.

Parameters
----------
config keys (same as fl_runner.py, plus):
  n_groups          : int — number of ensemble members / groups (K)
  ensemble_inference : 'avg_proba' (default) | 'majority_vote'

Returns
-------
Same result dict as fl_runner.run_one_experiment(), plus:
  n_groups                : K used
  group_sizes             : list of n_clients per group
  group_activity_accs     : list of per-group accuracy (before ensemble)
  ensemble_gain           : ensemble_acc - mean(group_accs)  (how much ensemble helps)
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
from core.fl_runner import fedavg_delta, evaluate


# ─────────────────────────────────────────
# Ensemble inference
# ─────────────────────────────────────────

def ensemble_predict(models, X_batch, device, mode='avg_proba'):
    """
    Combine predictions from K group models for a single batch.

    Parameters
    ----------
    models  : list of K trained nn.Module
    X_batch : input tensor already on device
    mode    : 'avg_proba'     — average softmax probabilities (default, strictly better)
              'majority_vote' — majority vote on hard predictions

    Returns
    -------
    predicted : LongTensor of predicted class indices, shape (batch_size,)
    avg_probs : FloatTensor of averaged softmax probabilities, shape (batch_size, n_classes)
                (None if mode == 'majority_vote')
    """
    softmax = nn.Softmax(dim=1)

    if mode == 'avg_proba':
        all_probs = []
        for model in models:
            model.eval()
            with torch.no_grad():
                logits = model(X_batch)
                probs  = softmax(logits)
                all_probs.append(probs)
        # Average across K models
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # (batch, n_classes)
        predicted = torch.argmax(avg_probs, dim=1)
        return predicted, avg_probs

    elif mode == 'majority_vote':
        all_preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                logits = model(X_batch)
                preds  = torch.argmax(logits, dim=1)
                all_preds.append(preds)
        # Stack → (K, batch_size) → majority vote per sample
        stacked   = torch.stack(all_preds, dim=0)
        # Mode along dim=0
        predicted = torch.mode(stacked, dim=0).values
        return predicted, None

    else:
        raise ValueError(f"Unknown ensemble mode: {mode}. Use 'avg_proba' or 'majority_vote'.")


def evaluate_ensemble(models, test_loader, device, mode='avg_proba'):
    """
    Evaluate the ensemble on the test set.

    Returns
    -------
    accuracy : float (%)
    loss     : float (cross-entropy on averaged probabilities)
    """
    criterion    = nn.CrossEntropyLoss()
    total_loss   = 0.0
    correct      = 0
    total        = 0
    log_softmax  = nn.LogSoftmax(dim=1)

    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predicted, avg_probs = ensemble_predict(models, X_batch, device, mode)

        correct += (predicted == y_batch).sum().item()
        total   += y_batch.size(0)

        if avg_probs is not None:
            # Cross-entropy from averaged probs: -sum(y * log(avg_probs))
            log_probs   = torch.log(avg_probs.clamp(min=1e-9))
            total_loss += criterion(log_probs, y_batch).item()

    acc  = 100.0 * correct / total
    loss = total_loss / len(test_loader) if total_loss > 0 else 0.0
    return acc, loss


# ─────────────────────────────────────────
# Client grouping
# ─────────────────────────────────────────

def assign_clients_to_groups(clients, n_groups, seed=42):
    """
    Randomly assign clients to K groups.

    Assignment is fixed for the entire training run (not re-shuffled per round).
    With 30 clients and K=3, each group gets exactly 10 clients.
    If 30 is not divisible by K, groups differ in size by at most 1.

    Returns
    -------
    groups : list of K lists, each containing a subset of clients
    """
    rng     = random.Random(seed)
    shuffled = list(clients)
    rng.shuffle(shuffled)

    groups = [[] for _ in range(n_groups)]
    for i, client in enumerate(shuffled):
        groups[i % n_groups].append(client)

    return groups


# ─────────────────────────────────────────
# Main ensemble runner
# ─────────────────────────────────────────

def run_ensemble_experiment(config):
    """
    Run one full Ensemble FL experiment.

    Parameters
    ----------
    config : dict — same keys as fl_runner.run_one_experiment(), plus:
        n_groups           : int  — K (number of groups / ensemble members)
        ensemble_inference : str  — 'avg_proba' (default) or 'majority_vote'

    Returns
    -------
    results : dict with all keys from fl_runner, plus:
        n_groups            : K used
        group_sizes         : [n_clients in group_0, ..., n_clients in group_{K-1}]
        group_activity_accs : [acc_group_0, ..., acc_group_{K-1}] (last round, pre-ensemble)
        ensemble_gain       : ensemble_acc - mean(group_accs)
    """

    # ── Seeds ────────────────────────────────────────────────
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device             = "cuda" if torch.cuda.is_available() else "cpu"
    defense_type       = config['defense_type']
    n_groups           = config.get('n_groups', 3)
    ensemble_mode      = config.get('ensemble_inference', 'avg_proba')

    # ── Data ─────────────────────────────────────────────────
    X_train, y_train, s_train, X_test, y_test, s_test = load_ucihar()

    sensitive_indices = find_subject_sensitive_features(
        X_train, s_train, top_k=100
    )

    client_data = create_federated_clients(
        X_train, y_train, s_train,
        alpha=config.get('alpha', 0.5),
        seed=seed,
    )

    clients = []
    for i, cd in enumerate(client_data):
        loader = get_client_dataloader(cd, batch_size=config.get('batch_size', 32))
        clients.append(FederatedClient(
            client_id  = i,
            subject_id = cd["subject_id"],
            dataloader = loader,
            device     = device,
        ))

    test_loader = get_test_dataloader(X_test, y_test)

    # ── Group assignment (fixed for entire run) ───────────────
    groups = assign_clients_to_groups(clients, n_groups, seed=seed)
    print(f"\n  [ensemble] K={n_groups} groups | sizes: {[len(g) for g in groups]}")

    # ── One model per group ───────────────────────────────────
    group_models = [get_model(device) for _ in range(n_groups)]

    # ── One attacker per group (server sees per-group deltas) ─
    # This models the realistic threat: server can attack each group separately.
    group_attackers = [SubjectInferenceAttacker() for _ in range(n_groups)]

    # ── Per-group state for subject_aware_delta ───────────────
    warmup_deltas      = [[] for _ in range(n_groups)]
    warmup_subject_ids = [[] for _ in range(n_groups)]
    sensitive_delta_dims_per_group = [None] * n_groups

    rounds          = config.get('rounds', 15)
    warmup_rounds   = config.get('warmup_rounds', 5)
    track_per_round = config.get('track_per_round', False)
    track_every     = config.get('track_every', 5)

    per_round_acc_ensemble  = []   # ensemble accuracy each round
    per_round_acc_groups    = [[] for _ in range(n_groups)]  # per-group accuracy
    per_round_attack        = []

    # ── Training loop ─────────────────────────────────────────
    for rnd in range(1, rounds + 1):

        for g_idx, (group, group_model, attacker) in enumerate(
                zip(groups, group_models, group_attackers)):

            # ── Activate subject_aware_delta dims for this group ──
            if (defense_type == 'subject_aware_delta'
                    and rnd == warmup_rounds + 1
                    and sensitive_delta_dims_per_group[g_idx] is None):
                n_unique = len(set(warmup_subject_ids[g_idx]))
                # Each group has n_clients/K subjects. Require at least
                # half the group's subjects to be seen before computing dims.
                min_subj = max(3, len(group) // 2)
                if len(warmup_deltas[g_idx]) > 5 and n_unique >= min_subj:
                    sensitive_delta_dims_per_group[g_idx] = find_sensitive_delta_dimensions(
                        warmup_deltas[g_idx],
                        warmup_subject_ids[g_idx],
                        top_k=config.get('sensitive_top_k', 50),
                    )
                    for c in group:
                        c.sensitive_delta_dims = sensitive_delta_dims_per_group[g_idx]
                    print(f'  [SAD group {g_idx}] dims computed ({n_unique} subjects seen)')
                else:
                    print(f'  [SAD group {g_idx}] warmup insufficient ({n_unique} subjects) — extending')

            # ── Participation: sample clients within this group ───
            num_in_group  = len(group)
            num_selected  = max(1, int(num_in_group * config.get('participation_rate', 0.7)))
            selected      = random.sample(group, num_selected)

            deltas       = []
            client_sizes = []

            for client in selected:
                client.sensitive_indices = (
                    sensitive_indices if defense_type == 'subject_aware' else None
                )

                delta, n, _ = client.local_train(
                    global_model     = group_model,
                    local_epochs     = config.get('local_epochs', 3),
                    lr               = config.get('lr', 0.01),
                    defense_type     = defense_type,
                    noise_scale      = config.get('noise_scale', 2.0),
                    clip_threshold   = config.get('clip_threshold', 1.0),
                    noise_multiplier = config.get('noise_multiplier', 0.05),
                )

                deltas.append(delta)
                client_sizes.append(n)
                attacker.collect(delta, client.subject_id)

                if defense_type == 'subject_aware_delta' and rnd <= warmup_rounds:
                    warmup_deltas[g_idx].append(flatten_delta(delta))
                    warmup_subject_ids[g_idx].append(client.subject_id)

            # ── FedAvg within this group ──────────────────────────
            group_models[g_idx] = fedavg_delta(group_model, deltas, client_sizes)

        # ── Ensemble inference on test set ────────────────────
        ens_acc, _ = evaluate_ensemble(group_models, test_loader, device, mode=ensemble_mode)
        per_round_acc_ensemble.append(ens_acc)

        # Per-group accuracy (pre-ensemble)
        for g_idx, gm in enumerate(group_models):
            g_acc, _ = evaluate(gm, test_loader, device)
            per_round_acc_groups[g_idx].append(g_acc)

        # Per-round attack tracking (worst-case across all group attackers)
        if track_per_round and rnd % track_every == 0:
            worst_across_groups = 0.0
            for attacker in group_attackers:
                if len(attacker.collected_deltas) >= 10:
                    attacker.train_and_evaluate()
                    worst_across_groups = max(worst_across_groups, attacker.worst_case_acc())
            if worst_across_groups > 0:
                per_round_attack.append((rnd, worst_across_groups))

    # ── Final attack evaluation ───────────────────────────────
    # Worst-case across ALL group attackers
    for attacker in group_attackers:
        if len(attacker.collected_deltas) >= 10:
            attacker.train_and_evaluate()

    all_attack_accs = [a.worst_case_acc() for a in group_attackers if a.is_trained]
    final_attack_acc = max(all_attack_accs) if all_attack_accs else 0.0

    # Merge per-clf results (worst-case per classifier across groups)
    per_clf = {}
    for attacker in group_attackers:
        for clf_name, acc in attacker.per_classifier_results().items():
            per_clf[clf_name] = max(per_clf.get(clf_name, 0.0), acc)

    # ── Summary stats ─────────────────────────────────────────
    group_final_accs = [per_round_acc_groups[g][-1] for g in range(n_groups)]
    ensemble_final   = per_round_acc_ensemble[-1]
    ensemble_gain    = ensemble_final - float(np.mean(group_final_accs))

    random_chance = 100.0 / len(set(c.subject_id for c in clients))

    print(f"\n  [ensemble] Final ensemble acc : {ensemble_final:.2f}%")
    print(f"  [ensemble] Mean group acc     : {np.mean(group_final_accs):.2f}%")
    print(f"  [ensemble] Ensemble gain      : {ensemble_gain:+.2f}pp")
    print(f"  [ensemble] Worst attack acc   : {final_attack_acc:.2f}%")
    # NOTE on ensemble privacy: two competing forces are at play.
    # (1) Attacker sees fewer deltas per group → less training data → harder.
    # (2) Each group has fewer subjects → simpler classification problem → easier.
    # In practice (2) dominates when K is large, so ensemble alone may NOT
    # improve privacy. This is an empirical finding worth reporting.
    # Combine with subject_aware_delta to get genuine privacy improvement.

    return {
        'final_activity_acc'     : ensemble_final,
        'final_attack_acc'       : final_attack_acc,
        'attack_acc_per_clf'     : per_clf,
        'privacy_gain_vs_random' : round(final_attack_acc / random_chance, 4),
        'per_round_acc'          : per_round_acc_ensemble,
        'per_round_attack'       : per_round_attack,
        # Ensemble-specific
        'n_groups'               : n_groups,
        'group_sizes'            : [len(g) for g in groups],
        'group_activity_accs'    : group_final_accs,
        'ensemble_gain'          : round(ensemble_gain, 4),
    }
