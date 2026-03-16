# Installation

## Requirements

- **GPU**: NVIDIA GPU with CUDA 12.4+
- **Build Tools**: CMake (≥ 3.20), GCC, NVCC
- **Python**: Python 3.8+ (for Python bindings)

## C++ Executable

### Build from Source

Clone the repository and compile the project using CMake:

```bash
git clone https://github.com/Lhongpei/PDHCG-II.git
cd PDHCG-II
cmake -S . -B build
cmake --build build --clean-first
```

This will create the solver binary at `./build/bin/pdhcg`.

### Specifying CUDA Compiler

If your system has multiple CUDA versions or the default nvcc is outdated, explicitly specify the path to your CUDA compiler:

```bash
# Replace '/your/path/to/nvcc' with the actual path, e.g., /usr/local/cuda-12.6/bin/nvcc
CUDACXX=/your/path/to/nvcc cmake -S . -B build
cmake --build build --clean-first
```

## Python Package

### From PyPI (Recommended)

```bash
pip install pdhcg
```

### From Source

```bash
git clone https://github.com/Lhongpei/PDHCG-II.git
cd PDHCG-II
pip install .
```

### Development Installation

For development with editable install:

```bash
git clone https://github.com/Lhongpei/PDHCG-II.git
cd PDHCG-II
pip install -e ".[test]"
```

## Verification

### C++ Executable

```bash
./build/bin/pdhcg --help
```

### Python Package

```python
import pdhcg
print(pdhcg.__version__)
```

## Pre-commit Hooks (For Contributors)

To ensure code quality before committing, install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This automatically formats Python (Ruff), C/C++ (clang-format), and checks spelling when you commit.
