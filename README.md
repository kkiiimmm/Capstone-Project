# AI-Based Transmit Power Control for Multi-Router IoT Networks

This graduation project studies AI-based transmit power control in a multi-router IoT network.

## System Model

- There are `K` routers and `K` IoT devices.
- Router `k` communicates with device `k`.
- `H[k, m]` is the channel gain from router `m` to device `k`.
- `H` is already a power gain/pathloss-scale value, not an amplitude (so it is
  used directly in SINR/capacity and is **not** squared).
- SINR:

```text
SINR_k = P_k * H[k,k] / (N + sum_{m != k} P_m * H[k,m])
```

- Sum capacity:

```text
C_sum = sum_k log2(1 + SINR_k)
```

## Files

- `01_generate_H.py`: Generates random 2D router/device positions and distance-based channel gain matrices.
- `02_fullsearch_power.py`: Python full search that creates optimal power labels (vectorized). Equivalent to `fullsearch_power.m`; use either one.
- `fullsearch_power.m`: MATLAB full search that creates the same optimal power labels.
- `03_train_dnn.py`: Trains a PyTorch DNN to classify one power candidate per router (supervised).
- `04_train_unsupervised_dnn.py`: Trains a PyTorch DNN by directly maximizing sum capacity (no labels).
- `05_train_dqn.py`: Trains a one-step DQN (contextual bandit) to select one complete power-vector action.
- `06_compare_methods.py`: Summarizes full-search, DNN, DQN, random, and all-max results.
- `07_ber_simulation.py`: BPSK bit-error-rate simulation for each method's chosen power vector.
- `08_shannon_throughput.py`: Ideal-code Shannon throughput for each method's chosen power vector.
- `visualize_network.m`: Creates MATLAB figures for presentation slides.
- `data/`: Generated `.dat` files (H, positions, full-search labels).
- `results/`: Output directory created by the training/analysis scripts.

## Generated Data

Run:

```bash
python 01_generate_H.py
```

The script creates:

- `data/H_train.dat`
- `data/H_test.dat`
- `data/positions_train.dat`
- `data/positions_test.dat`

Each row in `H_train.dat` and `H_test.dat` is one flattened `K x K` channel gain matrix
(row-major: `H11 H12 ... H1K H21 ... HKK`).

Each row in `positions_train.dat` and `positions_test.dat` stores the matching positions for one sample:

```text
router_0_x router_0_y ... router_K-1_x router_K-1_y device_0_x device_0_y ... device_K-1_x device_K-1_y
```

## Main Parameters

The main constants are defined near the top of `01_generate_H.py`:

- `K` (default `6`)
- `AREA_SIZE`
- `PATHLOSS_EXPONENT`
- `MIN_DISTANCE`
- `TRAIN_SAMPLE_COUNT`
- `TEST_SAMPLE_COUNT`
- `RANDOM_SEED`

## Discrete Transmit-Power Candidates

All methods (full search, DNNs, DQN) share the same discrete candidate set,
defined as `POWER_CANDIDATES` at the top of each script:

```text
[0.0, 0.25, 0.5, 0.75, 1.0]
```

`0.0` means the router is turned **OFF**. Including OFF is important: in an
interference-limited network, the sum-capacity-optimal solution often turns off
weak / strongly-interfering links and serves only a clean subset, so the
power-control problem also becomes a link-selection (on/off) problem. With this
default data, the full-search optimum keeps only about `1.4` of the `6` routers
ON per sample. The `ON_THRESHOLD = 1e-5` used in the analysis cleanly separates
OFF (`0.0`) from ON (`>= 0.25`).

> If you change `POWER_CANDIDATES`, you must regenerate the labels
> (`02_fullsearch_power.py` or `fullsearch_power.m`) because the supervised DNN
> and every capacity ratio are computed against those labels.

### DNN power output

For `K = 6` and `5` candidates, each network outputs `6 x 5 = 30` logits. The
logits are reshaped to `[batch, 6, 5]`, then softmax is applied separately over
the five candidates for each router, so each router selects one candidate
independently.

- The **supervised DNN** (`03`) uses cross entropy against the full-search candidate index.
- The **unsupervised DNN** (`04`) uses no labels. It forms a differentiable
  *soft* (expected) power from the softmax probabilities and maximizes sum
  capacity directly. Note the soft power is a continuous relaxation in
  `[0, 1]`, so its capacity can slightly exceed the discrete full-search optimum
  (`soft_ratio > 1` is possible). The fair, apples-to-apples comparison is the
  **hard** (argmax) power, which is restricted to the discrete candidates.
- The **DQN** (`05`) treats one complete power vector as one action. With `5`
  candidates and `K = 6`, the action space is `5^6 = 15625`. It uses
  `GAMMA = 0` (a one-step contextual bandit) and learns from the sum-capacity
  reward of the selected action. Because the action space is large, the DQN
  typically needs many episodes and is the weakest of the learned methods;
  increase `EPISODES` if you want better coverage.

## Baselines

`06`, `07`, `08` also report two non-AI baselines:

- **All-max**: every router at the maximum candidate power `1.0`.
- **Random**: each router picks a candidate uniformly at random (drawn per sample).

## Run Order

1. Generate channel gain matrices:

```bash
python 01_generate_H.py
```

2. Generate full-search labels (pick **one**):

```bash
python 02_fullsearch_power.py
```

or, in MATLAB:

```matlab
fullsearch_power
```

3. Train and evaluate the supervised DNN:

```bash
python 03_train_dnn.py
```

4. Train and evaluate the unsupervised/direct-optimization DNN:

```bash
python 04_train_unsupervised_dnn.py
```

5. Train and evaluate the one-step DQN:

```bash
python 05_train_dqn.py
```

6. Compare all methods:

```bash
python 06_compare_methods.py
```

7. (Optional) BER and Shannon-throughput analysis:

```bash
python 07_ber_simulation.py
python 08_shannon_throughput.py
```

8. Create MATLAB visualization figures:

```matlab
visualize_network
```
