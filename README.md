# PDHCG-II: A GPU-Accelerated Solver for Quadratic Programming

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Publication](https://img.shields.io/badge/DOI-10.1287/ijoc.2024.0983-B31B1B.svg)](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983)
[![arXiv](https://img.shields.io/badge/arXiv-2405.16160-b31b1b.svg)](https://arxiv.org/abs/2405.16160)

**PDHCG** is a high-performance, GPU-accelerated implementation of the Primal-Dual Hybrid Gradient (PDHG) algorithm designed for solving large-scale Quadratic Programming (QP) problems. This solver is developed by Hongpei Li, Yicheng Huang, Huikang Liu, Dongdong Ge, and Yinyu Ye.

For a detailed explanation of the methodology, please refer to our paper: [A GPU-Based Primal-Dual Hybrid Gradient Method for Quadratic Programming](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983).

---

## Problem Formulation

PDHCG solves quadratic programs in the following form, which allow a flexibile input of quadratic objective matrix, a sparse component and a dense low-rank component:

```math
\begin{aligned}
\min_{x} \quad & \frac{1}{2}x^\top (Q + R^\top R) x + c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x \le u_v.
\end{aligned}
```


## Installation (C++ Executable)

To use the standalone C++ solver, you must compile the project using CMake.

### Requirements
* **GPU:** NVIDIA GPU with CUDA 12.4+.
* **Build Tools:** CMake (≥ 3.20), GCC, NVCC.

### Build from Source
Clone the repository and compile the project using CMake.
```bash
git clone https://github.com/Lhongpei/PDHCGv2-C.git
cd PDHCGv2-C
cmake -B build
cmake --build build --clean-first
```
This will create the solver binary at `./build/bin/pdhcg`.


##  Usage (C++ Executable)

Run the solver from the command line:

```bash
./build/bin/pdhcg <MPS_FILE> <OUTPUT_DIR> [OPTIONS]
```

### Command Line Arguments

**Positional Arguments:**

1. `<MPS_FILE>`: Path to the input QP (supports `.mps`, `.qps`, and `.mps.gz`).
2. `<OUTPUT_DIR>`: Directory where solution files will be saved.

**Solver Parameters:**
| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `-h`, `--help` | `flag` | Display the help message. | N/A |
| `-v`, `--verbose` | `flag` | Enable verbose logging. | `false` |
| `--time_limit` | `double` | Time limit in seconds. | `3600.0` |
| `--iter_limit` | `int` | Iteration limit. | `2147483647` |
| `--eps_opt` | `double` | Relative optimality tolerance. | `1e-4` |
| `--eps_feas` | `double` | Relative feasibility tolerance. | `1e-4` |

---

<!-- ## Citation

If you use PDHCG in your research, please cite our paper:

```bibtex
@article{Li2024,
  author  = {Li, Hongpei and Huang, Yicheng and Liu, Huikang and Ge, Dongdong and Ye, Yinyu},
  title   = {A GPU-Based Primal-Dual Hybrid Gradient Method for Quadratic Programming},
  journal = {INFORMS Journal on Computing},
  year    = {2024},
  doi     = {10.1287/ijoc.2024.0983},
  URL     = {https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0983}
}
```

--- -->

## Python Interface

PDHCG provides a user-friendly Python interface that allows you to define, solve, and analyze QP problems using familiar libraries like NumPy and SciPy.

For detailed instructions on how to use the Python interface, including installation, modeling, and examples, please see the [Python Interface README](./python/README.md).

### Quick Example in Python

```python
import numpy as np
import scipy.sparse as sp
from pdhcg import Model, PDHCG

# Example: minimize 0.5 * x'Qx + c'x
# subject to l <= A x <= u,  lb <= x <= ub
Q = sp.csc_matrix([[1.0, -1.0], [-1.0, 2.0]])
c = np.array([-2.0, -6.0])
A = sp.csc_matrix([[1.0, 1.0], [-1.0, 2.0], [2.0, 1.0]])
l = np.array([-np.inf, -np.inf, -np.inf])
u = np.array([2.0, 2.0, 3.0])
lb = np.zeros(2)
ub = np.array([np.inf, np.inf])

# Create QP model
m = Model(objective_matrix=Q,
          objective_vector=c,
          constraint_matrix=A,
          constraint_lower_bound=l,
          constraint_upper_bound=u,
          variable_lower_bound=lb,
          variable_upper_bound=ub)

# Solve
m.optimize()

# Print results
print("Status:", m.Status)
print("Objective:", m.ObjVal)
```
## License

Copyright 2024-2026 Hongpei Li.

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
