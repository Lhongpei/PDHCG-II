# D-PDLP: A First-Order LP Solver Accelerated on Multiple GPUs

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2601.07628-B31B1B.svg)](https://arxiv.org/pdf/2601.07628)

**D-PDLP** (Distributed PDLP) is a high-performance, distributed implementation of the Primal-Dual Hybrid Gradient (PDHG) algorithm designed for solving massive-scale Linear Programming (LP) problems on multi-GPU systems.

By leveraging 2D Grid Partitioning, D-PDLP scales the first-order PDHG method across GPU clusters, efficiently harnessing the aggregate computational power and memory of multiple devices. This implementation is built upon [cuPDLPx](https://github.com/Lhongpei/cuPDLPx), a GPU-accelerated LP solver described in [cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming](https://arxiv.org/abs/2507.14051).

For a detailed explanation of the methodology, please refer to our paper: [Beyond Single-GPU: Scaling PDLP to Distributed Multi-GPU Systems](https://arxiv.org/pdf/2601.07628).

---

## Problem Formulation

Consistent with cuPDLPx, D-PDLP solves linear programs in the standard form:

```math
\begin{aligned}
\min_{x} \quad & c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x \le u_v.
\end{aligned}
```

## Installation

To use the solver, you must compile the project using CMake.

### Requirements
* **GPU:** NVIDIA GPU with CUDA 12.4+.
* **Build Tools:** CMake (≥ 3.20), GCC, NVCC.
* **Distributed Tolls:** MPI, NCCL.
### Build from Source
Clone the repository and compile the project using CMake.
```bash
git clone git@github.com:Lhongpei/D-PDLP.git
cd D-PDLP
cmake -B build
cmake --build build --clean-first
```
This will create the solver binary at `./build/cupdlpx-dist`.


##  Usage

The executable supports both single-GPU and multi-GPU distributed modes. It automatically detects the mode based on the MPI launcher.

### 1. Single GPU Mode

Run the solver directly without MPI to use a single GPU.

```bash
./build/cupdlpx-dist <MPS_FILE> <OUTPUT_DIR> [OPTIONS]

```

### 2. Distributed Multi-GPU Mode

Use `mpirun` to launch the solver across multiple GPUs.

```bash
mpirun -n <NUM_GPU> ./build/cupdlpx-dist <MPS_FILE> <OUTPUT_DIR> [OPTIONS]

```

### Command Line Arguments

**Positional Arguments:**

1. `<MPS_FILE>`: Path to the input LP (supports `.mps` and `.mps.gz`).
2. `<OUTPUT_DIR>`: Directory where solution files will be saved.

**Distributed Options:**

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--grid_size <r>,<c>` | `string` | 2D Grid topology (Rows x Cols) | Auto-detect |
| `--partition_method` | `string` | Partitioning strategy: `uniform` or `nnz`. | `nnz` |
| `--permute_method` | `string` | Matrix permutation: `none`, `random`, or `block`. | `none` |

**Solver Parameters:**
| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `-h`, `--help` | `flag` | Display the help message. | N/A |
| `-v`, `--verbose` | `flag` | Enable verbose logging. | `false` |
| `--time_limit` | `double` | Time limit in seconds. | `3600.0` |
| `--iter_limit` | `int` | Iteration limit. | `2147483647` |
| `--eps_opt` | `double` | Relative optimality tolerance. | `1e-4` |
| `--eps_feas` | `double` | Relative feasibility tolerance. | `1e-4` |
| `--eps_infeas_detect` | `double` | Infeasibility detection tolerance. | `1e-10` |
| `--l_inf_ruiz_iter` | `int` | Iterations for L-inf Ruiz rescaling| `10` |
| `--no_pock_chambolle` | `flag` | Disable Pock-Chambolle rescaling | `enabled` |
| `--pock_chambolle_alpha` | `float` | Value for Pock-Chambolle alpha | `1.0` |
| `--no_bound_obj_rescaling` | `flag` | Disable bound objective rescaling | `enabled` |
| `--eval_freq` | `int` | Termination evaluation frequency | `200` |
| `--sv_max_iter` | `int` | Max iterations for singular value estimation | `5000` |
| `--sv_tol` | `float` | Tolerance for singular value estimation | `1e-4` |

---

## Output Artifacts

Upon successful completion, the solver generates three files in the specified output directory:

1. **`<PROBLEM>_summary.txt`**: Scalar metrics (Time, Iterations, Primal/Dual values, Residuals).
2. **`<PROBLEM>_primal_solution.txt`**: The full primal solution vector (one float per line).
3. **`<PROBLEM>_dual_solution.txt`**: The full dual solution vector (one float per line).

---

## Citation

If you use this software or method in your research, please cite our paper:

```bibtex
@article{li2026beyond,
  title={Beyond Single-GPU: Scaling PDLP to Distributed Multi-GPU Systems},
  author={Li, Hongpei and Huang, Yicheng and Liu, Huikang and Ge, Dongdong and Ye, Yinyu},
  journal={arXiv preprint arXiv:2601.07628},
  year={2026}
}

```

---

## License

Copyright 2025-2026 Hongpei Li, Haihao Lu.

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
