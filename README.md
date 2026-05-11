# AI-Based Transmit Power Control for Multi-Router IoT Networks

This graduation project studies AI-based transmit power control in a multi-router IoT network.

## System Model

- There are `K` routers and `K` IoT devices.
- Router `k` communicates with device `k`.
- `H[k, m]` is the channel gain from router `m` to device `k`.
- `H` is already a power gain/pathloss-scale value, not an amplitude.
- Later SINR calculations should use:

```text
SINR_k = P_k * H[k,k] / (N + sum_{m != k} P_m * H[k,m])
```

## Initial Files

- `01_generate_H.py`: Generates random 2D router/device positions and distance-based channel gain matrices.
- `fullsearch_power.m`: Runs MATLAB full search to create optimal power labels.
- `03_train_dnn.py`: Trains a PyTorch DNN to classify one power candidate per router.
- `04_train_unsupervised_dnn.py`: Trains a PyTorch DNN by directly maximizing sum capacity.
- `05_train_dqn.py`: Trains a one-step DQN to select one complete power-vector action.
- `06_compare_methods.py`: Summarizes full-search, DNN, DQN, random, and all-max results.
- `visualize_network.m`: Creates MATLAB figures for presentation slides.
- `data/`: Output directory for generated `.dat` files.
- `results/`: Output directory for trained DNN results.

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

Each row in `H_train.dat` and `H_test.dat` is one flattened `K x K` channel gain matrix.

Each row in `positions_train.dat` and `positions_test.dat` stores the matching positions for one sample:

```text
router_0_x router_0_y ... router_K-1_x router_K-1_y device_0_x device_0_y ... device_K-1_x device_K-1_y
```

## Main Parameters

The main constants are defined near the top of `01_generate_H.py`:

- `K`
- `AREA_SIZE`
- `PATHLOSS_EXPONENT`
- `MIN_DISTANCE`
- `TRAIN_SAMPLE_COUNT`
- `TEST_SAMPLE_COUNT`
- `RANDOM_SEED`

The default value is `K = 6`.

## DNN Power Output

The DNNs use the same discrete transmit-power candidates as the MATLAB full search:

```text
[1e-10, 0.33, 0.66, 0.99]
```

For `K = 6`, each network outputs `6 x 4 = 24` logits. The logits are reshaped to `[batch, 6, 4]`, then softmax is applied separately over the four candidates for each router. This means each router selects one candidate independently.

The supervised DNN uses cross entropy against the full-search candidate index. The unsupervised DNN does not use training labels; it uses the softmax probabilities to form differentiable expected powers and maximizes the sum capacity directly.

The DQN treats one complete power vector as one action. It does not use full-search labels for training; it learns from the sum capacity reward for the selected action.

## Run Order

1. Generate channel gain matrices:

```bash
python 01_generate_H.py
```

2. Generate full-search labels in MATLAB:

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

7. Create MATLAB visualization figures:

```matlab
visualize_network
```
