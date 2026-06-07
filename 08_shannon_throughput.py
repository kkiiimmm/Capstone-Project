from pathlib import Path

import numpy as np


# Easy-to-modify constants
K = 6
INPUT_DIM = K * K
POWER_CANDIDATES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
NOISE_POWER = 1e-6
RANDOM_SEED = 42
CHANNEL_USES_PER_SAMPLE = 10000  # matches BITS_PER_ROUTER in 07 for fair comparison

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def load_methods():
    """Load each method's chosen transmit-power vector per test sample."""
    H_test = np.loadtxt(DATA_DIR / "H_test.dat")
    label_test = np.loadtxt(DATA_DIR / "label_test.dat")
    dnn_result = np.loadtxt(RESULTS_DIR / "dnn_result.dat")
    unsup_result = np.loadtxt(RESULTS_DIR / "unsupervised_result.dat")
    dqn_result = np.loadtxt(RESULTS_DIR / "dqn_result.dat")

    if H_test.shape[1] != INPUT_DIM:
        raise ValueError(f"H_test must have {INPUT_DIM} columns.")
    if label_test.shape[1] != K + 1:
        raise ValueError(f"label_test must have {K + 1} columns.")

    n = min(
        H_test.shape[0],
        label_test.shape[0],
        dnn_result.shape[0],
        unsup_result.shape[0],
        dqn_result.shape[0],
    )

    powers = {
        "Full-search": label_test[:n, 1 : 1 + K],
        "Supervised DNN": dnn_result[:n, 4 : 4 + K],
        "Unsupervised DNN hard": unsup_result[:n, 12 : 12 + K],
        "DQN": dqn_result[:n, 5 : 5 + K],
    }
    return H_test[:n], powers, n


def add_baselines(powers, n, rng):
    """Append non-AI baselines: random pick and all-max.

    Random powers are drawn per sample (one rng.choice of size K for each of
    the n samples), matching the convention used in 04/05/06 so the baselines
    stay comparable across scripts.
    """
    random_power = np.zeros((n, K))
    for i in range(n):
        random_power[i] = rng.choice(POWER_CANDIDATES, size=K)
    powers["Random"] = random_power
    powers["All-max"] = np.full((n, K), 1.0)
    return powers


def shannon_rate_per_device(H, P, noise_power):
    """
    Per-device Shannon rate in bits per channel use.

    Assumes an ideal capacity-achieving code so every link can transmit
    log2(1 + SINR_k) reliable bits per channel use.
    """
    desired = P * np.diag(H)
    received = H.dot(P)
    interference = received - desired
    sinr = desired / (noise_power + interference)
    if np.any(sinr < 0.0):
        raise ValueError("SINR contains negative values.")
    return np.log2(1.0 + sinr)


def run_simulation(H_test, powers, n):
    """Compute Shannon throughput per sample for every method."""
    methods = list(powers.keys())
    metrics = {
        m: {
            "rate_per_device": np.zeros((n, K)),
            "sum_rate": np.zeros(n),
            "reliable_bits": np.zeros(n),
        }
        for m in methods
    }

    for i in range(n):
        H = H_test[i].reshape(K, K)
        for m in methods:
            P = powers[m][i]
            rates = shannon_rate_per_device(H, P, NOISE_POWER)
            metrics[m]["rate_per_device"][i] = rates
            metrics[m]["sum_rate"][i] = rates.sum()
            metrics[m]["reliable_bits"][i] = rates.sum() * CHANNEL_USES_PER_SAMPLE

        if (i + 1) % 500 == 0:
            print(f"Computed {i + 1}/{n} samples")

    return methods, metrics


def print_and_save_summary(methods, metrics):
    print()
    print(
        f"{'Method':<24} {'Avg sum-rate':>14} {'Median sum-rate':>16} "
        f"{'Avg reliable bits':>18}"
    )
    print("-" * 76)

    lines = [
        "method,avg_sum_rate_bpcu,median_sum_rate_bpcu,"
        "avg_reliable_bits_per_sample\n"
    ]
    for m in methods:
        d = metrics[m]
        avg_rate = d["sum_rate"].mean()
        median_rate = np.median(d["sum_rate"])
        avg_bits = d["reliable_bits"].mean()
        print(
            f"{m:<24} {avg_rate:>14.6e} {median_rate:>16.6e} "
            f"{avg_bits:>18.2f}"
        )
        lines.append(f"{m},{avg_rate},{median_rate},{avg_bits}\n")

    summary_path = RESULTS_DIR / "shannon_summary.csv"
    summary_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nSaved {summary_path}")


def save_per_sample(methods, metrics):
    n = metrics[methods[0]]["sum_rate"].shape[0]
    columns = [np.arange(1, n + 1).reshape(-1, 1)]
    columns.extend(metrics[m]["sum_rate"].reshape(-1, 1) for m in methods)
    np.savetxt(RESULTS_DIR / "shannon_sum_rate_per_sample.dat", np.hstack(columns))

    columns = [np.arange(1, n + 1).reshape(-1, 1)]
    columns.extend(metrics[m]["reliable_bits"].reshape(-1, 1) for m in methods)
    np.savetxt(
        RESULTS_DIR / "shannon_reliable_bits_per_sample.dat", np.hstack(columns)
    )

    print(f"Saved {RESULTS_DIR / 'shannon_sum_rate_per_sample.dat'}")
    print(f"Saved {RESULTS_DIR / 'shannon_reliable_bits_per_sample.dat'}")
    print(f"Column order: sample_index, " + ", ".join(methods))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    H_test, powers, n = load_methods()
    powers = add_baselines(powers, n, rng)

    print(
        f"Computing Shannon throughput assuming an ideal capacity-achieving code. "
        f"Channel uses per sample: {CHANNEL_USES_PER_SAMPLE}."
    )
    print(f"Test samples: {n}, methods: {len(powers)}")

    methods, metrics = run_simulation(H_test, powers, n)
    print_and_save_summary(methods, metrics)
    save_per_sample(methods, metrics)


if __name__ == "__main__":
    main()
