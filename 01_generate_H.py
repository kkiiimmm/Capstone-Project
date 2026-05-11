from pathlib import Path

import numpy as np


# Easy-to-modify simulation constants
K = 6
AREA_SIZE = 100.0
PATHLOSS_EXPONENT = 3.0
MIN_DISTANCE = 1.0
TRAIN_SAMPLE_COUNT = 10000
TEST_SAMPLE_COUNT = 2000
RANDOM_SEED = 42
OUTPUT_DIR = Path("data")


def generate_positions(rng, sample_count, k, area_size):
    """Generate router and IoT device positions in a square area."""
    router_positions = rng.uniform(0.0, area_size, size=(sample_count, k, 2))
    device_positions = rng.uniform(0.0, area_size, size=(sample_count, k, 2))
    return router_positions, device_positions


def compute_channel_gain(router_positions, device_positions, pathloss_exponent, min_distance):
    """
    Compute H[k, m], the power gain from router m to device k.

    H is already a power gain/pathloss-scale value, so later SINR calculations
    should use H directly and should not square it.
    """
    displacement = device_positions[:, :, np.newaxis, :] - router_positions[:, np.newaxis, :, :]
    distances = np.linalg.norm(displacement, axis=-1)
    distances = np.maximum(distances, min_distance)

    return 1.0 / (distances ** pathloss_exponent)


def flatten_positions(router_positions, device_positions):
    """Store router coordinates first, then IoT device coordinates."""
    sample_count = router_positions.shape[0]
    return np.concatenate(
        [
            router_positions.reshape(sample_count, -1),
            device_positions.reshape(sample_count, -1),
        ],
        axis=1,
    )


def check_channel_gain(H, sample_count, k):
    """Run basic sanity checks on the generated channel gain matrices."""
    expected_shape = (sample_count, k, k)
    if H.shape != expected_shape:
        raise ValueError(f"Expected H shape {expected_shape}, got {H.shape}")
    if not np.all(np.isfinite(H)):
        raise ValueError("H contains non-finite values")
    if not np.all(H >= 0.0):
        raise ValueError("H contains negative values")


def save_dataset(name, H, router_positions, device_positions, output_dir):
    """Save flattened H matrices and matching positions."""
    H_flat = H.reshape(H.shape[0], -1)
    positions_flat = flatten_positions(router_positions, device_positions)

    H_path = output_dir / f"H_{name}.dat"
    positions_path = output_dir / f"positions_{name}.dat"

    np.savetxt(H_path, H_flat)
    np.savetxt(positions_path, positions_flat)

    loaded_H = np.loadtxt(H_path)
    loaded_positions = np.loadtxt(positions_path)

    if loaded_H.shape != H_flat.shape:
        raise ValueError(f"Loaded {H_path} has shape {loaded_H.shape}, expected {H_flat.shape}")
    if loaded_positions.shape != positions_flat.shape:
        raise ValueError(
            f"Loaded {positions_path} has shape {loaded_positions.shape}, "
            f"expected {positions_flat.shape}"
        )


def generate_dataset(rng, sample_count):
    """Generate positions and the corresponding K x K channel gain matrix H."""
    router_positions, device_positions = generate_positions(rng, sample_count, K, AREA_SIZE)
    H = compute_channel_gain(router_positions, device_positions, PATHLOSS_EXPONENT, MIN_DISTANCE)
    check_channel_gain(H, sample_count, K)
    return H, router_positions, device_positions


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    H_train, routers_train, devices_train = generate_dataset(rng, TRAIN_SAMPLE_COUNT)
    H_test, routers_test, devices_test = generate_dataset(rng, TEST_SAMPLE_COUNT)

    save_dataset("train", H_train, routers_train, devices_train, OUTPUT_DIR)
    save_dataset("test", H_test, routers_test, devices_test, OUTPUT_DIR)

    print(f"Saved training data: {TRAIN_SAMPLE_COUNT} samples")
    print(f"Saved test data: {TEST_SAMPLE_COUNT} samples")
    print(f"Each H row has {K * K} values for one flattened {K} x {K} matrix")
    print(f"Each positions row has {4 * K} values: routers first, then IoT devices")


if __name__ == "__main__":
    main()
