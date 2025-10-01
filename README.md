# NoStaLISA

**Non-Stationary LISA Analysis**: A tutorial on covariance matrices for modulated toy models in gravitational wave data analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This repository contains an educational tutorial exploring the construction and analysis of covariance matrices for toy models with time-dependent modulation. This is particularly relevant for gravitational wave data analysis with space-based detectors like LISA, where signals may exhibit amplitude modulation due to detector orientation changes, orbital motion, or source evolution.

### What You'll Learn

- How to construct modulated gravitational wave signals
- Computing noise power spectral densities (PSDs) for space-based detectors
- Building covariance matrices in both time and frequency domains
- Understanding the relationship between time-domain windowing and frequency-domain covariance
- Efficient computational methods for handling large covariance matrices using FFT-based convolution

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Installing uv

If you don't have `uv` installed, you can install it using one of these methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more installation options, visit [uv's documentation](https://github.com/astral-sh/uv).

### Setting Up the Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lorenzsp/NoStaLISA.git
   cd NoStaLISA
   ```

2. **Create a virtual environment with uv:**
   ```bash
   uv venv
   ```

3. **Activate the virtual environment:**
   
   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```
   
   **Windows:**
   ```powershell
   .venv\Scripts\activate
   ```

4. **Install required packages:**
   ```bash
   uv pip install numpy scipy matplotlib tqdm jupyter jupyterlab
   ```

   Or install from the requirements file if you create one (see below).

### Alternative: One-Command Setup with pyproject.toml

If you prefer the modern Python packaging approach with `pyproject.toml`:

```bash
# Create venv and install in one command
uv venv && source .venv/bin/activate && uv pip install -e .
```

### Alternative: Using Requirements File

Or use the traditional `requirements.txt`:

```bash
uv pip install -r requirements.txt
```
### Verify Installation

After installation, verify everything is working:

```bash
python -c "import numpy, scipy, matplotlib, tqdm; print('✓ All packages installed successfully!')"
```

## Quick Start

1. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

2. **Open the tutorial notebook:**
   Navigate to `toy_model_modulation_td.ipynb` in the Jupyter Lab interface.

3. **Run the cells sequentially:**
   Execute each cell in order to follow the tutorial from basic concepts to advanced techniques.

## Repository Structure

```
Non_Stationary_Work/
├── README.md                        # This file
├── toy_model_modulation_td.ipynb   # Main tutorial notebook
├── toy_models/                      # Additional toy models (if any)
└── requirements.txt                 # Python dependencies (optional)
```

## Tutorial Contents

The notebook is organized into 10 main sections:

1. **Introduction & Overview** - Learning objectives and background
2. **Function Definitions** - Signal generation and noise modeling functions
3. **Signal Parameters Setup** - Time series construction and modulation
4. **Noise Covariance Construction** - Building covariance matrices
5. **DFT via Linear Algebra** - Understanding matrix representations
6. **Windowed Covariance** - Core analysis with visualizations
7. **Numerical Cleanup** - Optimization techniques
8. **Efficient Computation** - FFT-based methods (O(N² log N))
9. **Summary & Conclusions** - Key takeaways
10. **Optional Exercises** - Hands-on practice problems

## Key Features

- **Comprehensive Documentation**: All functions include detailed NumPy-style docstrings
- **Educational Visualizations**: Rich plots with annotations explaining key features
- **Two Computational Approaches**: 
  - Direct method (O(N³)) for pedagogical clarity
  - Efficient FFT-based method (O(N² log N)) for practical applications
- **Real-World Applications**: Based on LISA noise models and gravitational wave analysis

## Dependencies

- **NumPy**: Numerical computing and array operations
- **SciPy**: Signal processing, FFTs, and sparse linear algebra
- **Matplotlib**: Plotting and visualization
- **tqdm**: Progress bars for long computations
- **Jupyter/JupyterLab**: Interactive notebook environment

## Performance Notes

The tutorial includes computation of large covariance matrices. For optimal performance:

- The notebook uses power-of-2 FFT sizes for efficiency
- Sparse matrix representations are discussed for large datasets
- Memory usage is monitored and reported throughout

For very large `N` values (>16384), consider:
- Using the efficient FFT-based method only
- Implementing sparse matrix operations
- Running on a machine with sufficient RAM (8GB+ recommended)

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest improvements or additional examples
- Submit pull requests with enhancements

## License

[Add your license information here]

## Contact

For questions or feedback, please open an issue on GitHub or contact the repository maintainers.

## Acknowledgments

This work is part of the LISA mission preparation and gravitational wave data analysis research.

---

**Happy Learning!** 🌊📊🔬