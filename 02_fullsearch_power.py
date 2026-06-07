from pathlib import Path

import numpy as np


# Easy-to-modify constants
K = 6
INPUT_DIM = K * K
POWER_CANDIDATES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
NOISE_POWER = 1e-6

DATA_DIR = Path("data")


def generate_power_vectors(power_candidates, k):
    """Return all candidate^k combinations as rows of a (num_actions, k) array."""
    num_candidates = len(power_candidates)
    num_actions = num_candidates ** k
    vectors = np.zeros((num_actions, k), dtype=np.float64)

    for index in range(num_actions):
        value = index
        for router in range(k):
            vectors[index, router] = power_candidates[value % num_candidates]
            value //= num_candidates

    return vectors


def batch_sum_capacity(H, power_vectors, noise_power):
    """
    Vectorized sum-capacity over every candidate power vector for one H.

    H[k, m] is the power gain from router m to device k. Signal at device k is
    P_k * H[k, k]; interference is sum over m != k of P_m * H[k, m].
    """
    diag = np.diag(H)
    received = power_vectors @ H.T
    desired = power_vectors * diag[np.newaxis, :]
    interference = received - desired

    sinr = desired / (noise_power + interference)
    if np.any(sinr < 0.0):
        raise ValueError("SINR contains negative values.")

    capacities = np.sum(np.log2(1.0 + sinr), axis=1)
    if not np.all(np.isfinite(capacities)):
        raise ValueError("Capacity contains non-finite values.")

    return capacities


def process_dataset(input_path, output_path, power_vectors):
    H_rows = np.loadtxt(input_path)
    if H_rows.shape[1] != INPUT_DIM:
        raise ValueError(f"{input_path} must have {INPUT_DIM} columns.")

    n = H_rows.shape[0]
    labels = np.zeros((n, K + 1))

    print(f"Processing {input_path}: {n} samples")
    for i in range(n):
        H = H_rows[i].reshape(K, K)
        capacities = batch_sum_capacity(H, power_vectors, NOISE_POWER)
        best = int(np.argmax(capacities))
        labels[i, 0] = capacities[best]
        labels[i, 1:] = power_vectors[best]

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}")

    np.savetxt(output_path, labels)
    print(f"Saved {output_path}")


def main():
    power_vectors = generate_power_vectors(POWER_CANDIDATES, K)
    if not np.all(np.isin(power_vectors, POWER_CANDIDATES)):
        raise ValueError("Generated power vectors must use only the candidates.")

    process_dataset(DATA_DIR / "H_train.dat", DATA_DIR / "label_train.dat", power_vectors)
    process_dataset(DATA_DIR / "H_test.dat", DATA_DIR / "label_test.dat", power_vectors)


if __name__ == "__main__":
    main()
