from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Easy-to-modify constants
K = 6
INPUT_DIM = K * K
POWER_CANDIDATES = np.array([0.25, 0.5, 0.75, 1.0])
NUM_POWER_CANDIDATES = len(POWER_CANDIDATES)
OUTPUT_DIM = K * NUM_POWER_CANDIDATES
NOISE_POWER = 1e-6
EPSILON = 1e-30
RANDOM_SEED = 42

BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
PRINT_EVERY = 50

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


class PowerControlDNN(nn.Module):
    """Classify one discrete transmit-power candidate for each router."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, output_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return reshape_logits(logits)


def reshape_logits(logits):
    if logits.shape[1] != OUTPUT_DIM:
        raise ValueError(f"Logits must have {OUTPUT_DIM} columns.")

    logits = logits.view(-1, K, NUM_POWER_CANDIDATES)
    if logits.shape[1:] != (K, NUM_POWER_CANDIDATES):
        raise ValueError(f"Logits must reshape to [batch_size, {K}, {NUM_POWER_CANDIDATES}].")

    return logits


def load_data():
    H_train = np.loadtxt(DATA_DIR / "H_train.dat")
    H_test = np.loadtxt(DATA_DIR / "H_test.dat")
    label_train = np.loadtxt(DATA_DIR / "label_train.dat")
    label_test = np.loadtxt(DATA_DIR / "label_test.dat")

    if H_train.shape[1] != INPUT_DIM or H_test.shape[1] != INPUT_DIM:
        raise ValueError(f"H files must have {INPUT_DIM} columns.")
    if label_train.shape[1] != K + 1 or label_test.shape[1] != K + 1:
        raise ValueError(f"Label files must have {K + 1} columns.")

    H_train, label_train = match_debug_label_count(H_train, label_train, "train")
    H_test, label_test = match_debug_label_count(H_test, label_test, "test")

    return H_train, H_test, label_train, label_test


def match_debug_label_count(H, labels, name):
    """Use only labeled H rows when MATLAB full search was run in DEBUG_MODE."""
    if labels.shape[0] < H.shape[0]:
        print(
            f"Warning: {name} labels have {labels.shape[0]} rows, "
            f"but H has {H.shape[0]} rows. Using the first {labels.shape[0]} H rows."
        )
        H = H[: labels.shape[0]]

    if labels.shape[0] != H.shape[0]:
        raise ValueError(f"{name} H rows and label rows do not match.")

    return H, labels


def normalize_H(H_train, H_test):
    """Log-scale small channel gains, then standardize using train statistics."""
    X_train = np.log10(H_train + EPSILON)
    X_test = np.log10(H_test + EPSILON)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-12)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train.astype(np.float32), X_test.astype(np.float32), mean, std


def powers_to_candidate_indices(power_labels):
    """Convert full-search power labels to nearest discrete candidate indices."""
    distances = np.abs(power_labels[:, :, np.newaxis] - POWER_CANDIDATES[np.newaxis, np.newaxis, :])
    indices = np.argmin(distances, axis=2)
    reconstructed = POWER_CANDIDATES[indices]

    if not np.allclose(power_labels, reconstructed, rtol=1e-5, atol=1e-8):
        print("Warning: some power labels were mapped to the nearest candidate.")

    return indices.astype(np.int64)


def probabilities_from_logits(logits):
    probabilities = torch.softmax(logits, dim=2)
    sums = probabilities.sum(dim=2)

    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise ValueError("Softmax probabilities must sum to 1 for each router.")

    return probabilities


def indices_to_powers(indices):
    powers = POWER_CANDIDATES[indices]
    if not np.all(np.isin(powers, POWER_CANDIDATES)):
        raise ValueError("Predicted hard powers must be valid power candidates.")
    return powers


def compute_sinr(H, P, noise_power):
    """
    Compute SINR for all IoT devices.

    H[k, m] is the power gain from router m to device k. H is already a
    power gain/pathloss-scale value, so it is used directly and is not squared.
    """
    desired_signal = P * np.diag(H)
    received_power = H.dot(P)
    interference = received_power - desired_signal
    sinr = desired_signal / (noise_power + interference)

    if np.any(sinr < 0.0):
        raise ValueError("SINR contains negative values.")

    return sinr


def compute_sum_capacity(H, P, noise_power):
    sinr = compute_sinr(H, P, noise_power)
    capacity = np.sum(np.log2(1.0 + sinr))

    if not np.isfinite(capacity):
        raise ValueError("Capacity is not finite.")

    return capacity


def train_model(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for X_batch, target_indices in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(
                logits.reshape(-1, NUM_POWER_CANDIDATES),
                target_indices.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        average_loss = total_loss / len(train_loader.dataset)

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS}, train cross entropy: {average_loss:.6e}")


def evaluate_model(model, X_test, H_test, label_test, target_test_indices):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test)
        logits = model(X_tensor)
        probabilities = probabilities_from_logits(logits)
        predicted_indices = torch.argmax(probabilities, dim=2).cpu().numpy()

    predicted_power = indices_to_powers(predicted_indices)

    fullsearch_capacity = label_test[:, 0]
    dnn_capacity = np.zeros(H_test.shape[0])
    for i in range(H_test.shape[0]):
        H = H_test[i].reshape(K, K)
        dnn_capacity[i] = compute_sum_capacity(H, predicted_power[i], NOISE_POWER)

    capacity_ratio = dnn_capacity / fullsearch_capacity
    per_router_accuracy = np.mean(predicted_indices == target_test_indices)
    exact_vector_accuracy = np.mean(np.all(predicted_indices == target_test_indices, axis=1))

    result = np.column_stack(
        [
            np.arange(1, H_test.shape[0] + 1),
            fullsearch_capacity,
            dnn_capacity,
            capacity_ratio,
            predicted_power,
        ]
    )

    print(f"Per-router candidate accuracy: {per_router_accuracy:.6e}")
    print(f"Exact full-vector accuracy: {exact_vector_accuracy:.6e}")
    print(f"Average full-search capacity: {np.mean(fullsearch_capacity):.6e}")
    print(f"Average DNN capacity: {np.mean(dnn_capacity):.6e}")
    print(f"Average capacity ratio: {np.mean(capacity_ratio):.6e}")

    return result


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    H_train, H_test, label_train, label_test = load_data()
    X_train, X_test, mean, std = normalize_H(H_train, H_test)

    target_train_indices = powers_to_candidate_indices(label_train[:, 1:])
    target_test_indices = powers_to_candidate_indices(label_test[:, 1:])

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(target_train_indices),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PowerControlDNN(INPUT_DIM, OUTPUT_DIM)
    train_model(model, train_loader)

    result = evaluate_model(model, X_test, H_test, label_test, target_test_indices)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "K": K,
            "power_candidates": POWER_CANDIDATES,
            "input_mean": mean,
            "input_std": std,
            "epsilon": EPSILON,
        },
        RESULTS_DIR / "trained_model.pt",
    )

    np.savetxt(RESULTS_DIR / "dnn_result.dat", result)
    print(f"Saved {RESULTS_DIR / 'trained_model.pt'}")
    print(f"Saved {RESULTS_DIR / 'dnn_result.dat'}")


if __name__ == "__main__":
    main()
