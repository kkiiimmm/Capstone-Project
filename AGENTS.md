# Project Instructions

This repository is for a graduation project about AI-based transmit power control in a multi-router IoT network.

Do not mix this project with the user's separate academic paper.
Do not include progressive image transmission, MIMO sub-band selection, spectral efficiency, spatial multiplexing, or Power Set Softmax.

System model:
- K routers and K IoT devices.
- Router k communicates with device k.
- H[k, m] is the channel gain from router m to device k.
- H is already a power gain/pathloss-scale value.
- Do not square H in SINR or capacity calculations.

SINR:
SINR_k = P_k * H[k,k] / (N + sum_{m != k} P_m * H[k,m])

Capacity:
C_sum = sum_k log2(1 + SINR_k)

Coding style:
- Keep code simple and readable.
- Prefer functions over giant scripts.
- Use clear filenames.
- Add comments explaining communication-system meaning.
- Keep constants easy to modify.