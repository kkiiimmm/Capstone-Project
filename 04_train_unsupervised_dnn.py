from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Easy-to-modify constants
K = 6
INPUT_DIM = K * K
POWER_CANDIDATES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
NUM_POWER_CANDIDATES = len(POWER_CANDIDATES)
OUTPUT_DIM = K * NUM_POWER_CANDIDATES
NOISE_POWER = 1e-6
EPSILON = 1e-30
RANDOM_SEED = 42

BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
PRINT_EVERY = 25

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


class UnsupervisedPowerDNN(nn.Module):
    """Predict per-router logits over discrete transmit-power candidates."""

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
    """Load H data and test labels. Training does not use full-search labels."""
    H_train = np.loadtxt(DATA_DIR / "H_train.dat").astype(np.float32)
    H_test = np.loadtxt(DATA_DIR / "H_test.dat").astype(np.float32)
    label_test = np.loadtxt(DATA_DIR / "label_test.dat").astype(np.float32)

    if H_train.shape[1] != INPUT_DIM or H_test.shape[1] != INPUT_DIM:
        raise ValueError(f"H_train and H_test must have {INPUT_DIM} columns.")
    if label_test.shape[1] != K + 1:
        raise ValueError(f"label_test must have {K + 1} columns.")

    if label_test.shape[0] < H_test.shape[0]:
        print(
            f"Warning: label_test has {label_test.shape[0]} rows, "
            f"but H_test has {H_test.shape[0]} rows. "
            f"Using the first {label_test.shape[0]} H_test rows."
        )
        H_test = H_test[: label_test.shape[0]]

    if label_test.shape[0] != H_test.shape[0]:
        raise ValueError("H_test rows and label_test rows do not match.")

    return H_train, H_test, label_test


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


def probabilities_from_logits(logits):
    probabilities = torch.softmax(logits, dim=2)
    sums = probabilities.sum(dim=2)

    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise ValueError("Softmax probabilities must sum to 1 for each router.")

    return probabilities


def soft_powers_from_logits(logits, power_candidates):
    probabilities = probabilities_from_logits(logits)
    return torch.sum(probabilities * power_candidates.view(1, 1, -1), dim=2)


def hard_powers_from_logits(logits):
    probabilities = probabilities_from_logits(logits)
    indices = torch.argmax(probabilities, dim=2).cpu().numpy()
    powers = POWER_CANDIDATES[indices]

    if not np.all(np.isin(powers, POWER_CANDIDATES)):
        raise ValueError("Predicted hard powers must be valid power candidates.")

    return powers


def torch_sum_capacity(H_batch, P_batch, noise_power):
    """
    Differentiable sum-capacity calculation used as the training objective.

    This is unsupervised/direct optimization because full-search power labels are
    not used in the loss. The DNN maximizes this communication equation directly.
    """
    batch_size = H_batch.shape[0]
    H = H_batch.view(batch_size, K, K)

    # Desired signal at device k comes from its paired router k.
    direct_gain = torch.diagonal(H, dim1=1, dim2=2)
    desired_signal = P_batch * direct_gain

    # Interference at device k is the received power from every other router.
    received_power = torch.bmm(H, P_batch.unsqueeze(2)).squeeze(2)
    interference = received_power - desired_signal

    # H is already a power gain/pathloss-scale value, so it is not squared.
    sinr = desired_signal / (noise_power + interference + 1e-20)
    capacity = torch.sum(torch.log2(1.0 + sinr), dim=1)

    if torch.any(sinr < -1e-8):
        raise ValueError("SINR contains negative values.")
    if not torch.all(torch.isfinite(capacity)):
        raise ValueError("Capacity contains non-finite values.")

    return capacity


def numpy_sum_capacity(H_flat, P, noise_power):
    """NumPy capacity calculation for evaluation and simple baselines."""
    H = H_flat.reshape(K, K)

    desired_signal = P * np.diag(H)
    received_power = H.dot(P)
    interference = received_power - desired_signal

    sinr = desired_signal / (noise_power + interference + 1e-20)
    if np.any(sinr < -1e-8):
        raise ValueError("SINR contains negative values.")

    capacity = np.sum(np.log2(1.0 + sinr))
    if not np.isfinite(capacity):
        raise ValueError("Capacity is not finite.")

    return capacity


def train_model(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    power_candidates = torch.tensor(POWER_CANDIDATES, dtype=torch.float32)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_capacity = 0.0

        for X_batch, H_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            P_soft = soft_powers_from_logits(logits, power_candidates)
            capacity = torch_sum_capacity(H_batch, P_soft, NOISE_POWER)

            # Minimize negative capacity to maximize network sum capacity.
            loss = -capacity.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_capacity += capacity.mean().item() * X_batch.size(0)

        average_loss = total_loss / len(train_loader.dataset)
        average_capacity = total_capacity / len(train_loader.dataset)

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(
                f"Epoch {epoch:4d}/{EPOCHS}, "
                f"negative capacity loss: {average_loss:.6e}, "
                f"train capacity: {average_capacity:.6e}"
            )


def evaluate_model(model, X_test, H_test, label_test):
    model.eval()
    power_candidates = torch.tensor(POWER_CANDIDATES, dtype=torch.float32)

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test)
        logits = model(X_tensor)
        P_soft = soft_powers_from_logits(logits, power_candidates).cpu().numpy()
        P_hard = hard_powers_from_logits(logits)

    if np.any(P_soft < 0.0) or np.any(P_soft > 1.0):
        raise ValueError("Soft predicted powers must be between 0 and 1.")

    fullsearch_capacity = label_test[:, 0]
    soft_capacity = np.zeros(H_test.shape[0], dtype=np.float32)
    hard_capacity = np.zeros(H_test.shape[0], dtype=np.float32)

    for i in range(H_test.shape[0]):
        soft_capacity[i] = numpy_sum_capacity(H_test[i], P_soft[i], NOISE_POWER)
        hard_capacity[i] = numpy_sum_capacity(H_test[i], P_hard[i], NOISE_POWER)

    soft_ratio = soft_capacity / fullsearch_capacity
    hard_ratio = hard_capacity / fullsearch_capacity

    result = np.column_stack(
        [
            np.arange(1, H_test.shape[0] + 1),
            fullsearch_capacity,
            soft_capacity,
            soft_ratio,
            hard_capacity,
            hard_ratio,
            P_soft,
            P_hard,
        ]
    )

    print(f"Average full-search capacity: {np.mean(fullsearch_capacity):.6e}")
    print(f"Average unsupervised soft capacity: {np.mean(soft_capacity):.6e}")
    print(f"Average unsupervised soft ratio: {np.mean(soft_ratio):.6e}")
    print(f"Average unsupervised hard capacity: {np.mean(hard_capacity):.6e}")
    print(f"Average unsupervised hard ratio: {np.mean(hard_ratio):.6e}")

    return result


def evaluate_baselines(H_test, label_test):
    fullsearch_capacity = label_test[:, 0]

    all_max_power = np.full(K, 1.0, dtype=np.float32)
    all_max_capacity = np.array(
        [numpy_sum_capacity(H_test[i], all_max_power, NOISE_POWER) for i in range(H_test.shape[0])]
    )

    rng = np.random.default_rng(RANDOM_SEED)
    random_capacity = np.zeros(H_test.shape[0], dtype=np.float32)
    for i in range(H_test.shape[0]):
        random_power = rng.choice(POWER_CANDIDATES, size=K)
        random_capacity[i] = numpy_sum_capacity(H_test[i], random_power, NOISE_POWER)

    print(f"Average all-max capacity: {np.mean(all_max_capacity):.6e}")
    print(f"Average all-max ratio: {np.mean(all_max_capacity / fullsearch_capacity):.6e}")
    print(f"Average random capacity: {np.mean(random_capacity):.6e}")
    print(f"Average random ratio: {np.mean(random_capacity / fullsearch_capacity):.6e}")


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    H_train, H_test, label_test = load_data()
    X_train, X_test, mean, std = normalize_H(H_train, H_test)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(H_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UnsupervisedPowerDNN(INPUT_DIM, OUTPUT_DIM)
    train_model(model, train_loader)

    result = evaluate_model(model, X_test, H_test, label_test)
    evaluate_baselines(H_test, label_test)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "K": K,
            "power_candidates": POWER_CANDIDATES,
            "input_mean": mean,
            "input_std": std,
            "epsilon": EPSILON,
        },
        RESULTS_DIR / "unsupervised_model.pt",
    )

    np.savetxt(RESULTS_DIR / "unsupervised_result.dat", result)
    print(f"Saved {RESULTS_DIR / 'unsupervised_model.pt'}")
    print(f"Saved {RESULTS_DIR / 'unsupervised_result.dat'}")


if __name__ == "__main__":
    main()
