from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


# Easy-to-modify constants
K = 6
INPUT_DIM = K * K
POWER_CANDIDATES = np.array([0.25, 0.5, 0.75, 1.0])
NUM_POWER_CANDIDATES = len(POWER_CANDIDATES)
NUM_ACTIONS = NUM_POWER_CANDIDATES ** K
NOISE_POWER = 1e-6
EPSILON = 1e-30
RANDOM_SEED = 42

EPISODES = 50000
BATCH_SIZE = 128
BUFFER_CAPACITY = 50000
WARMUP_STEPS = 1000
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 40000
GAMMA = 0.0
PRINT_EVERY = 1000

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


class ReplayBuffer:
    """Replay buffer for one-step DQN transitions: state, action, reward."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards = zip(*(self.buffer[i] for i in indices))
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
        )


class DQN(nn.Module):
    """Maps normalized channel state to one Q-value per power-vector action."""

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
        q_values = self.net(x)
        if q_values.shape[1] != NUM_ACTIONS:
            raise ValueError(f"DQN output shape must be [batch_size, {NUM_ACTIONS}].")
        return q_values


def generate_action_power_table(power_candidates, k):
    """Return all possible complete transmit-power vectors."""
    num_candidates = len(power_candidates)
    num_actions = num_candidates ** k
    table = np.zeros((num_actions, k), dtype=np.float64)

    for action_index in range(num_actions):
        value = action_index
        for router_index in range(k):
            candidate_index = value % num_candidates
            table[action_index, router_index] = power_candidates[candidate_index]
            value //= num_candidates

    if table.shape != (NUM_ACTIONS, K):
        raise ValueError(f"action_power_table must have shape [{NUM_ACTIONS}, {K}].")
    if not np.all(np.isin(table, power_candidates)):
        raise ValueError("Every action power must be one of the power candidates.")

    return table


def load_data():
    H_train = np.loadtxt(DATA_DIR / "H_train.dat")
    H_test = np.loadtxt(DATA_DIR / "H_test.dat")
    label_test = np.loadtxt(DATA_DIR / "label_test.dat")

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


def compute_sinr(H, P, noise_power):
    """
    Compute SINR for all IoT devices.

    Desired signal for device k is from router k. Interference is received
    power from all other routers. H is already a power gain/pathloss-scale
    value, so it is used directly and is not squared.
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


def compute_reward(H, action_index, action_power_table):
    P = action_power_table[action_index]
    H_matrix = H.reshape(K, K)
    return compute_sum_capacity(H_matrix, P, NOISE_POWER)


def epsilon_by_episode(episode):
    fraction = min(episode / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def choose_action(model, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(NUM_ACTIONS)

    model.eval()
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def update_dqn(model, optimizer, replay_buffer):
    states, actions, rewards = replay_buffer.sample(BATCH_SIZE)

    q_values = model(states)
    q_pred = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

    # One-step contextual bandit: GAMMA = 0, so target_q is just reward.
    q_target = rewards
    loss = nn.functional.mse_loss(q_pred, q_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_dqn(X_train, H_train, action_power_table):
    model = DQN(INPUT_DIM, NUM_ACTIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    recent_rewards = deque(maxlen=PRINT_EVERY)
    recent_losses = deque(maxlen=PRINT_EVERY)

    for episode in range(1, EPISODES + 1):
        sample_index = np.random.randint(H_train.shape[0])
        state = X_train[sample_index]
        H_original = H_train[sample_index]

        epsilon = epsilon_by_episode(episode)
        action = choose_action(model, state, epsilon)
        reward = compute_reward(H_original, action, action_power_table)

        replay_buffer.push(state, action, reward)
        recent_rewards.append(reward)

        if len(replay_buffer) >= WARMUP_STEPS:
            model.train()
            loss = update_dqn(model, optimizer, replay_buffer)
            recent_losses.append(loss)

        if episode == 1 or episode % PRINT_EVERY == 0:
            average_reward = np.mean(recent_rewards)
            average_loss = np.mean(recent_losses) if recent_losses else 0.0
            print(
                f"Episode {episode:5d}/{EPISODES}, "
                f"epsilon: {epsilon:.3f}, "
                f"average reward: {average_reward:.6e}, "
                f"DQN loss: {average_loss:.6e}"
            )

    return model


def evaluate_model(model, X_test, H_test, label_test, action_power_table):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test)
        q_values = model(X_tensor)
        action_indices = torch.argmax(q_values, dim=1).cpu().numpy()

    fullsearch_capacity = label_test[:, 0]
    dqn_capacity = np.zeros(H_test.shape[0])
    dqn_power = action_power_table[action_indices]

    for i in range(H_test.shape[0]):
        dqn_capacity[i] = compute_reward(H_test[i], action_indices[i], action_power_table)

    capacity_ratio = dqn_capacity / fullsearch_capacity

    result = np.column_stack(
        [
            np.arange(1, H_test.shape[0] + 1),
            fullsearch_capacity,
            dqn_capacity,
            capacity_ratio,
            action_indices,
            dqn_power,
        ]
    )

    print(f"Average full-search capacity: {np.mean(fullsearch_capacity):.6e}")
    print(f"Average DQN capacity: {np.mean(dqn_capacity):.6e}")
    print(f"Average capacity ratio: {np.mean(capacity_ratio):.6e}")

    return result


def evaluate_baselines(H_test, label_test):
    fullsearch_capacity = label_test[:, 0]

    all_max_power = np.full(K, 0.99)
    all_max_capacity = np.zeros(H_test.shape[0])
    for i in range(H_test.shape[0]):
        all_max_capacity[i] = compute_sum_capacity(H_test[i].reshape(K, K), all_max_power, NOISE_POWER)

    rng = np.random.default_rng(RANDOM_SEED)
    random_capacity = np.zeros(H_test.shape[0])
    for i in range(H_test.shape[0]):
        random_power = rng.choice(POWER_CANDIDATES, size=K)
        random_capacity[i] = compute_sum_capacity(H_test[i].reshape(K, K), random_power, NOISE_POWER)

    print(f"Average random capacity: {np.mean(random_capacity):.6e}")
    print(f"Average all-max capacity: {np.mean(all_max_capacity):.6e}")
    print(f"Average random ratio: {np.mean(random_capacity / fullsearch_capacity):.6e}")
    print(f"Average all-max ratio: {np.mean(all_max_capacity / fullsearch_capacity):.6e}")


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    action_power_table = generate_action_power_table(POWER_CANDIDATES, K)
    H_train, H_test, label_test = load_data()
    X_train, X_test, mean, std = normalize_H(H_train, H_test)

    model = train_dqn(X_train, H_train, action_power_table)
    result = evaluate_model(model, X_test, H_test, label_test, action_power_table)
    evaluate_baselines(H_test, label_test)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "K": K,
            "power_candidates": POWER_CANDIDATES,
            "action_power_table": action_power_table,
            "input_mean": mean,
            "input_std": std,
            "epsilon": EPSILON,
            "gamma": GAMMA,
        },
        RESULTS_DIR / "dqn_model.pt",
    )

    np.savetxt(RESULTS_DIR / "dqn_result.dat", result)
    print(f"Saved {RESULTS_DIR / 'dqn_model.pt'}")
    print(f"Saved {RESULTS_DIR / 'dqn_result.dat'}")


if __name__ == "__main__":
    main()
