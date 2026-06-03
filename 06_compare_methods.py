from pathlib import Path

import numpy as np


K = 6
INPUT_DIM = K * K
NOISE_POWER = 1e-6
POWER_CANDIDATES = np.array([0.25, 0.5, 0.75, 1.0])
RANDOM_SEED = 42
ON_THRESHOLD = 1e-5

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def load_results():
    H_test = np.loadtxt(DATA_DIR / "H_test.dat")
    label_test = np.loadtxt(DATA_DIR / "label_test.dat")
    dnn_result = np.loadtxt(RESULTS_DIR / "dnn_result.dat")
    unsup_result = np.loadtxt(RESULTS_DIR / "unsupervised_result.dat")
    dqn_result = np.loadtxt(RESULTS_DIR / "dqn_result.dat")

    if H_test.shape[1] != INPUT_DIM:
        raise ValueError(f"H_test must have {INPUT_DIM} columns.")
    if label_test.shape[1] != K + 1:
        raise ValueError(f"label_test must have {K + 1} columns.")
    if dnn_result.shape[1] != 10:
        raise ValueError("dnn_result.dat must have 10 columns.")
    if unsup_result.shape[1] != 18:
        raise ValueError("unsupervised_result.dat must have 18 columns.")
    if dqn_result.shape[1] != 11:
        raise ValueError("dqn_result.dat must have 11 columns.")

    sample_count = min(
        H_test.shape[0],
        label_test.shape[0],
        dnn_result.shape[0],
        unsup_result.shape[0],
        dqn_result.shape[0],
    )

    if sample_count < H_test.shape[0]:
        print(f"Warning: using first {sample_count} samples because result lengths differ.")

    return (
        H_test[:sample_count],
        label_test[:sample_count],
        dnn_result[:sample_count],
        unsup_result[:sample_count],
        dqn_result[:sample_count],
    )


def compute_sinr(H, P, noise_power):
    """
    Compute SINR for all IoT devices.

    Desired signal for device k comes from router k. Interference is the
    received power from other routers. H is already a power gain/pathloss-scale
    value, so H is used directly and is not squared.
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


def compute_baseline_capacities(H_test):
    rng = np.random.default_rng(RANDOM_SEED)

    random_capacity = np.zeros(H_test.shape[0])
    allmax_capacity = np.zeros(H_test.shape[0])
    allmax_power = np.full(K, 0.99)

    for i in range(H_test.shape[0]):
        H = H_test[i].reshape(K, K)
        random_power = rng.choice(POWER_CANDIDATES, size=K)
        random_capacity[i] = compute_sum_capacity(H, random_power, NOISE_POWER)
        allmax_capacity[i] = compute_sum_capacity(H, allmax_power, NOISE_POWER)

    return random_capacity, allmax_capacity


def summarize_method(method_name, capacity, fullsearch_capacity):
    ratio = capacity / fullsearch_capacity
    return {
        "method": method_name,
        "average_capacity": np.mean(capacity),
        "average_ratio": np.mean(ratio),
        "median_ratio": np.median(ratio),
        "minimum_ratio": np.min(ratio),
        "maximum_ratio": np.max(ratio),
        "std_ratio": np.std(ratio),
    }


def print_summary_table(rows):
    headers = [
        "Method",
        "Avg Capacity",
        "Avg Ratio",
        "Median",
        "Min",
        "Max",
        "Std",
    ]
    print()
    print(
        f"{headers[0]:<24} {headers[1]:>14} {headers[2]:>12} "
        f"{headers[3]:>12} {headers[4]:>12} {headers[5]:>12} {headers[6]:>12}"
    )
    print("-" * 104)

    for row in rows:
        print(
            f"{row['method']:<24} "
            f"{row['average_capacity']:>14.6e} "
            f"{row['average_ratio']:>12.6e} "
            f"{row['median_ratio']:>12.6e} "
            f"{row['minimum_ratio']:>12.6e} "
            f"{row['maximum_ratio']:>12.6e} "
            f"{row['std_ratio']:>12.6e}"
        )


def save_summary_table(rows):
    path = RESULTS_DIR / "summary_table.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("method,average_capacity,average_ratio,median_ratio,minimum_ratio,maximum_ratio,std_ratio\n")
        for row in rows:
            f.write(
                f"{row['method']},{row['average_capacity']},{row['average_ratio']},"
                f"{row['median_ratio']},{row['minimum_ratio']},{row['maximum_ratio']},"
                f"{row['std_ratio']}\n"
            )
    print(f"Saved {path}")


def save_capacity_ratios(
    fullsearch_capacity,
    dnn_capacity,
    unsup_soft_capacity,
    unsup_hard_capacity,
    dqn_capacity,
    random_capacity,
    allmax_capacity,
):
    sample_count = fullsearch_capacity.shape[0]
    ratio_table = np.column_stack(
        [
            np.arange(1, sample_count + 1),
            np.ones(sample_count),
            dnn_capacity / fullsearch_capacity,
            unsup_soft_capacity / fullsearch_capacity,
            unsup_hard_capacity / fullsearch_capacity,
            dqn_capacity / fullsearch_capacity,
            random_capacity / fullsearch_capacity,
            allmax_capacity / fullsearch_capacity,
        ]
    )

    path = RESULTS_DIR / "method_capacity_ratios.dat"
    np.savetxt(path, ratio_table)
    print(f"Saved {path}")


def print_power_analysis(name, powers):
    average_power = np.mean(powers, axis=0)
    on_fraction = np.mean(powers > ON_THRESHOLD, axis=0)

    print()
    print(f"{name} power analysis")
    print(f"Average selected power per router: {format_vector(average_power)}")
    print(f"ON fraction per router:          {format_vector(on_fraction)}")


def print_dqn_action_analysis(dqn_result):
    action_indices = dqn_result[:, 4].astype(int)
    powers = dqn_result[:, 5:11]
    unique_actions, counts = np.unique(action_indices, return_counts=True)
    order = np.argsort(counts)[::-1]

    print()
    print("DQN action analysis")
    print(f"Unique actions selected: {len(unique_actions)}")
    print("Top 10 action indices and counts:")
    for index in order[:10]:
        print(f"  action {unique_actions[index]:4d}: {counts[index]}")

    print_power_analysis("DQN", powers)


def format_vector(values):
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    H_test, label_test, dnn_result, unsup_result, dqn_result = load_results()

    fullsearch_capacity = label_test[:, 0]
    dnn_capacity = dnn_result[:, 2]
    unsup_soft_capacity = unsup_result[:, 2]
    unsup_hard_capacity = unsup_result[:, 4]
    dqn_capacity = dqn_result[:, 2]
    random_capacity, allmax_capacity = compute_baseline_capacities(H_test)

    rows = [
        summarize_method("Full-search", fullsearch_capacity, fullsearch_capacity),
        summarize_method("Supervised DNN", dnn_capacity, fullsearch_capacity),
        summarize_method("Unsupervised DNN soft", unsup_soft_capacity, fullsearch_capacity),
        summarize_method("Unsupervised DNN hard", unsup_hard_capacity, fullsearch_capacity),
        summarize_method("DQN", dqn_capacity, fullsearch_capacity),
        summarize_method("Random", random_capacity, fullsearch_capacity),
        summarize_method("All-max", allmax_capacity, fullsearch_capacity),
    ]

    print_summary_table(rows)
    save_summary_table(rows)
    save_capacity_ratios(
        fullsearch_capacity,
        dnn_capacity,
        unsup_soft_capacity,
        unsup_hard_capacity,
        dqn_capacity,
        random_capacity,
        allmax_capacity,
    )

    print_dqn_action_analysis(dqn_result)
    print_power_analysis("Supervised DNN", dnn_result[:, 4:10])
    print_power_analysis("Unsupervised DNN hard", unsup_result[:, 12:18])


if __name__ == "__main__":
    main()
