"""
data_utils.py
=============
Loads UCI-HAR dataset and creates a realistic Non-IID federated split.

UCI-HAR has 30 subjects. Each subject becomes one client naturally.
This gives unequal client sizes automatically — just like real FL.

Then we use Dirichlet to also skew the activity labels per client,
because in real life different people do different activities more.

Dataset:
  - 561 features (accelerometer + gyroscope readings)
  - 6 activity classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
                        SITTING, STANDING, LAYING
  - 30 subjects (clients)
  - Each subject has a DIFFERENT number of samples (unequal!)
"""

import numpy as np
import os
import urllib.request
import zipfile
import torch
from torch.utils.data import DataLoader, TensorDataset

# Resolve the default data directory relative to this file's location,
# so experiments work regardless of which directory they're run from.
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_THIS_DIR, "data", "ucihar")


# ─────────────────────────────────────────────────────────
# Download and load UCI-HAR
# ─────────────────────────────────────────────────────────

def download_ucihar(data_dir=None):
    """Download and extract UCI-HAR if not already present."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = os.path.join(data_dir, "ucihar.zip")
    extract_path = os.path.join(data_dir)

    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(os.path.join(data_dir, "UCI HAR Dataset")):
        print("UCI-HAR already downloaded.")
        return os.path.join(data_dir, "UCI HAR Dataset")

    print("Downloading UCI-HAR dataset (~60MB)...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_path)
    os.remove(zip_path)
    print("Done.")
    return os.path.join(data_dir, "UCI HAR Dataset")


def load_ucihar(data_dir=None):
    """
    Load UCI-HAR dataset.

    Returns
    -------
    X_train : (7352, 561) float32
    y_train : (7352,)     int   labels 0-5
    s_train : (7352,)     int   subject IDs 1-30
    X_test  : (2947, 561)
    y_test  : (2947,)
    s_test  : (2947,)
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    base = download_ucihar(data_dir)

    def read_file(path):
        return np.loadtxt(path)

    # Features
    X_train = read_file(os.path.join(base, "train", "X_train.txt")).astype(np.float32)
    X_test  = read_file(os.path.join(base, "test",  "X_test.txt" )).astype(np.float32)

    # Activity labels (originally 1-6, we shift to 0-5)
    y_train = read_file(os.path.join(base, "train", "y_train.txt")).astype(int) - 1
    y_test  = read_file(os.path.join(base, "test",  "y_test.txt" )).astype(int) - 1

    # Subject IDs (1-30) — this is what makes each client unique
    s_train = read_file(os.path.join(base, "train", "subject_train.txt")).astype(int)
    s_test  = read_file(os.path.join(base, "test",  "subject_test.txt" )).astype(int)

    return X_train, y_train, s_train, X_test, y_test, s_test


# ─────────────────────────────────────────────────────────
# Realistic Non-IID split
# ─────────────────────────────────────────────────────────

def create_federated_clients(X_train, y_train, s_train, alpha=0.5, seed=42):
    """
    Create one client per subject with Dirichlet-skewed activity labels.

    Why unequal sizes naturally?
      UCI-HAR was collected from 30 real people in real conditions.
      Some people did more recordings than others.
      So subject 1 might have 340 samples, subject 2 might have 280.
      We preserve this — no artificial balancing.

    What does Dirichlet add on top?
      Even within a subject's data, we further skew which activities
      they appear to do more. This simulates real life where
      one person mostly walks and another mostly sits.

    Parameters
    ----------
    X_train : feature array
    y_train : activity labels (0-5)
    s_train : subject IDs (1-30)
    alpha   : Dirichlet concentration
              0.1 = very skewed activity distribution per client
              1.0 = mild skew
              100 = nearly uniform activities per client

    Returns
    -------
    clients : list of dicts, each with:
              'subject_id' : which real subject this is
              'X'          : features for this client
              'y'          : activity labels
              'n_samples'  : how many samples (UNEQUAL across clients)
    """
    np.random.seed(seed)

    subject_ids = np.unique(s_train)  # [1, 2, 3, ..., 30]
    num_classes = len(np.unique(y_train))  # 6 activities

    clients = []

    for subject in subject_ids:
        # Get all samples belonging to this subject
        mask = (s_train == subject)
        X_subj = X_train[mask]
        y_subj = y_train[mask]

        # Apply Dirichlet skew to activity labels
        # This resamples the subject's data with skewed activity proportions
        X_skewed, y_skewed = apply_dirichlet_skew(X_subj, y_subj, num_classes, alpha)

        clients.append({
            "subject_id": subject,
            "X": X_skewed,
            "y": y_skewed,
            "n_samples": len(y_skewed),
        })

    return clients


def apply_dirichlet_skew(X, y, num_classes, alpha):
    """
    Resample data so that class proportions follow a Dirichlet distribution.

    Example: if alpha=0.1, one client might end up with
    80% WALKING samples and almost nothing else.

    Steps:
      1. Draw proportions from Dirichlet(alpha)
      2. Decide how many samples to take from each class
      3. Randomly sample that many from each class
      4. Return the combined resampled data
    """
    # Draw class proportions from Dirichlet
    proportions = np.random.dirichlet(np.repeat(alpha, num_classes))

    total = len(y)
    counts = (proportions * total).astype(int)
    # Fix rounding so total stays the same
    counts[-1] = total - counts[:-1].sum()
    counts = np.maximum(counts, 0)  # no negatives

    X_out, y_out = [], []

    for cls in range(num_classes):
        cls_mask = (y == cls)
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]

        n_available = len(y_cls)
        n_want = counts[cls]

        if n_available == 0 or n_want == 0:
            continue

        # Sample with replacement if we want more than available
        replace = n_want > n_available
        chosen = np.random.choice(n_available, size=n_want, replace=replace)
        X_out.append(X_cls[chosen])
        y_out.append(y_cls[chosen])

    if len(X_out) == 0:
        return X, y  # fallback: return original if something went wrong

    return np.vstack(X_out), np.concatenate(y_out)


# ─────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────

def get_client_dataloader(client_dict, batch_size=32, shuffle=True):
    """
    Build a PyTorch DataLoader from a client dict.

    Parameters
    ----------
    client_dict : one entry from create_federated_clients()
    batch_size  : training batch size
    shuffle     : shuffle each epoch

    Returns
    -------
    DataLoader
    """
    X = torch.tensor(client_dict["X"], dtype=torch.float32)
    y = torch.tensor(client_dict["y"], dtype=torch.long)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_test_dataloader(X_test, y_test, batch_size=256):
    """Build a DataLoader for the global test set."""
    X = torch.tensor(X_test, dtype=torch.float32)
    y = torch.tensor(y_test, dtype=torch.long)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ─────────────────────────────────────────────────────────
# Printing utilities
# ─────────────────────────────────────────────────────────

ACTIVITY_NAMES = [
    "WALKING", "WALK_UP", "WALK_DN", "SITTING", "STANDING", "LAYING"
]

def print_client_distributions(clients):
    """Print each client's sample count and activity distribution."""
    num_classes = 6
    print(f"\n{'Client':<10} {'Subj':<6} {'Samples':<9}", end="")
    for name in ACTIVITY_NAMES:
        print(f"  {name[:7]:>7}", end="")
    print()
    print("-" * (10 + 6 + 9 + num_classes * 9))

    for i, client in enumerate(clients):
        y = client["y"]
        counts = [(y == c).sum() for c in range(num_classes)]
        print(f"Client {i:<4} {client['subject_id']:<6} {client['n_samples']:<9}", end="")
        for c in counts:
            print(f"  {c:>7}", end="")
        print()

    sizes = [c["n_samples"] for c in clients]
    print(f"\nTotal clients : {len(clients)}")
    print(f"Max samples   : {max(sizes)}  (client with most data)")
    print(f"Min samples   : {min(sizes)}  (client with least data)")
    print(f"Total samples : {sum(sizes)}")