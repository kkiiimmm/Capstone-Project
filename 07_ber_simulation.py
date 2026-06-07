from pathlib import Path

import numpy as np


# Easy-to-modify constants
K = 6
INPUT_DIM = K * K
POWER_CANDIDATES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
NOISE_POWER = 1e-6
RANDOM_SEED = 42
BITS_PER_ROUTER = 10000
ON_THRESHOLD = 1e-5

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


def simulate_active_links(H, P, bits_per_router, rng):
    """
    Simulate BPSK transmission for one channel realization.

    Each router m sends random +-1 symbols with amplitude sqrt(P_m * H[k,m]).
    Routers turned OFF (P_m = 0, i.e. P_m <= ON_THRESHOLD) transmit no bits and
    contribute no signal: their amplitude sqrt(0 * H) = 0, so they neither serve
    their own device nor interfere with any other device.

    Returns active_mask and bit_errors per router (errors meaningful only
    where active_mask is True, since OFF routers transmit no bits).
    """
    active_mask = P > ON_THRESHOLD
    amplitudes = np.sqrt(P[np.newaxis, :] * H)

    bits = rng.integers(0, 2, size=(K, bits_per_router))
    symbols = (2 * bits - 1).astype(np.float64)

    received = amplitudes @ symbols
    noise = rng.normal(0.0, np.sqrt(NOISE_POWER), size=(K, bits_per_router))
    received = received + noise

    decided_symbols = np.where(received >= 0.0, 1, -1)
    decided_bits = (decided_symbols + 1) // 2
    bit_errors = (decided_bits != bits).sum(axis=1)
    return active_mask, bit_errors


def run_simulation(H_test, powers, n, rng):
    """Run BPSK simulation for every method on every test sample."""
    methods = list(powers.keys())
    metrics = {
        m: {
            "active_count": np.zeros(n),
            "attempted_bits": np.zeros(n),
            "successful_bits": np.zeros(n),
            "ber_active": np.full(n, np.nan),
        }
        for m in methods
    }

    for i in range(n):
        H = H_test[i].reshape(K, K)
        for m in methods:
            P = powers[m][i]
            active_mask, bit_errors = simulate_active_links(
                H, P, BITS_PER_ROUTER, rng
            )

            num_active = int(active_mask.sum())
            attempted = num_active * BITS_PER_ROUTER
            active_errors = int(bit_errors[active_mask].sum()) if num_active else 0
            successful = attempted - active_errors

            metrics[m]["active_count"][i] = num_active
            metrics[m]["attempted_bits"][i] = attempted
            metrics[m]["successful_bits"][i] = successful
            if num_active > 0:
                metrics[m]["ber_active"][i] = active_errors / attempted

        if (i + 1) % 200 == 0:
            print(f"Simulated {i + 1}/{n} samples")

    return methods, metrics


def print_and_save_summary(methods, metrics):
    print()
    print(
        f"{'Method':<24} {'Avg active links':>18} "
        f"{'Avg success bits':>18} {'Avg active BER':>16}"
    )
    print("-" * 80)

    lines = [
        "method,avg_active_links,avg_attempted_bits,"
        "avg_successful_bits,avg_active_ber\n"
    ]
    for m in methods:
        d = metrics[m]
        avg_active = d["active_count"].mean()
        avg_attempted = d["attempted_bits"].mean()
        avg_success = d["successful_bits"].mean()
        valid = np.isfinite(d["ber_active"])
        avg_ber = d["ber_active"][valid].mean() if valid.any() else float("nan")

        print(
            f"{m:<24} {avg_active:>18.4f} "
            f"{avg_success:>18.2f} {avg_ber:>16.6e}"
        )
        lines.append(
            f"{m},{avg_active},{avg_attempted},{avg_success},{avg_ber}\n"
        )

    summary_path = RESULTS_DIR / "ber_summary.csv"
    summary_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nSaved {summary_path}")


def save_per_sample(methods, metrics):
    n = metrics[methods[0]]["active_count"].shape[0]

    success_cols = [np.arange(1, n + 1).reshape(-1, 1)]
    success_cols.extend(metrics[m]["successful_bits"].reshape(-1, 1) for m in methods)
    np.savetxt(RESULTS_DIR / "successful_bits_per_sample.dat", np.hstack(success_cols))

    ber_cols = [np.arange(1, n + 1).reshape(-1, 1)]
    ber_cols.extend(metrics[m]["ber_active"].reshape(-1, 1) for m in methods)
    np.savetxt(RESULTS_DIR / "ber_active_per_sample.dat", np.hstack(ber_cols))

    print(f"Saved {RESULTS_DIR / 'successful_bits_per_sample.dat'}")
    print(f"Saved {RESULTS_DIR / 'ber_active_per_sample.dat'}")
    print(f"Column order: sample_index, " + ", ".join(methods))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    H_test, powers, n = load_methods()
    powers = add_baselines(powers, n, rng)

    print(f"Simulating BPSK with {BITS_PER_ROUTER} bits per router per sample.")
    print(f"Test samples: {n}, methods: {len(powers)}")

    methods, metrics = run_simulation(H_test, powers, n, rng)
    print_and_save_summary(methods, metrics)
    save_per_sample(methods, metrics)


if __name__ == "__main__":
    main()
