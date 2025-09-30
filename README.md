# NoStaLISA

**Non-Stationary Noise Analysis for LISA**

A Python package for investigating non-stationary noise in LISA (Laser Interferometer Space Antenna) data.

## Overview

NoStaLISA provides tools for analyzing non-stationary noise in gravitational wave detector data, specifically designed for LISA. The package includes three main modules:

- **Injection Module**: Tools for injecting various types of non-stationary noise features
- **Fitting Module**: PSD (Power Spectral Density) estimation and fitting algorithms
- **Test Module**: Statistical tests for detecting and characterizing non-stationarity

## Installation

### From source

Clone the repository and install:

```bash
git clone https://github.com/lorenzsp/NoStaLISA.git
cd NoStaLISA
pip install -e .
```

### Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0

## Quick Start

### Injection Module

Inject various types of non-stationary features into your data:

```python
import numpy as np
from nostalisa.injection import NonStationaryNoiseInjector

# Create an injector
injector = NonStationaryNoiseInjector(sampling_frequency=100.0, duration=10.0)

# Generate some data
data = np.random.randn(1000)

# Inject a glitch
data_with_glitch = injector.inject_glitch(
    data, 
    time=5.0,           # Time in seconds
    amplitude=10.0,     # Amplitude
    width=0.1,          # Width in seconds
    form='gaussian'     # Options: 'gaussian', 'sine-gaussian', 'delta'
)

# Inject a data gap
data_with_gap = injector.inject_gap(
    data,
    start_time=3.0,
    gap_duration=1.0,
    fill_value=0.0
)

# Inject non-Gaussian noise
data_with_ng = injector.inject_non_gaussian(
    data,
    distribution='lognormal',  # Options: 'lognormal', 'exponential', 'cauchy'
    mean=0.0,
    sigma=1.0
)
```

### Fitting Module

Estimate and fit power spectral densities:

```python
from nostalisa.fitting import PSDEstimator, PSDFitter

# Create PSD estimator
estimator = PSDEstimator(sampling_frequency=100.0, method='welch')

# Estimate PSD
freqs, psd = estimator.estimate(data, nperseg=256)

# Fit a power law model to the PSD
fitter = PSDFitter()
params, covariance = fitter.fit(freqs, psd)

# Evaluate the fitted model
fitted_psd = fitter.evaluate_model(freqs, params)

# Estimate time-varying PSD (spectrogram)
freqs, times, spectrogram = estimator.estimate_time_varying(data)
```

### Test Module

Run statistical tests for non-stationarity:

```python
from nostalisa.test import NonStationarityTester

# Create tester
tester = NonStationarityTester(data, sampling_frequency=100.0)

# Run various tests
runs_results = tester.runs_test()
variance_results = tester.variance_ratio_test(n_windows=10)
mean_results = tester.mean_stationarity_test(n_windows=10)
kurtosis_results = tester.kurtosis_test()
spectral_results = tester.spectral_stationarity_test(n_segments=4)

# Check results
print(f"Runs test p-value: {runs_results['p_value']}")
print(f"Variance ratio: {variance_results['variance_ratio']}")
print(f"Mean stationarity p-value: {mean_results['p_value']}")
print(f"Kurtosis: {kurtosis_results['kurtosis']}")
```

## Module Documentation

### Injection Module (`nostalisa.injection`)

**Classes:**
- `NonStationaryNoiseInjector`: Main class for noise injection with methods:
  - `inject_glitch()`: Inject transient glitches
  - `inject_gap()`: Inject data gaps
  - `inject_non_gaussian()`: Inject non-Gaussian noise

**Functions:**
- `inject_glitch()`: Convenience function for glitch injection
- `inject_gap()`: Convenience function for gap injection
- `inject_non_gaussian()`: Convenience function for non-Gaussian noise

### Fitting Module (`nostalisa.fitting`)

**Classes:**
- `PSDEstimator`: PSD estimation using various methods
  - `estimate()`: Estimate PSD using Welch or periodogram method
  - `estimate_time_varying()`: Compute spectrogram
- `PSDFitter`: Fit parametric models to PSDs
  - `fit()`: Fit model to PSD data
  - `evaluate_model()`: Evaluate fitted model

**Functions:**
- `welch_psd()`: Compute PSD using Welch's method
- `periodogram_psd()`: Compute PSD using periodogram
- `fit_psd_model()`: Fit parametric model to PSD

### Test Module (`nostalisa.test`)

**Classes:**
- `NonStationarityTester`: Statistical tests for non-stationarity
  - `runs_test()`: Test for randomness
  - `variance_ratio_test()`: Test for time-varying variance
  - `mean_stationarity_test()`: Test for stationarity in mean
  - `kurtosis_test()`: Test for non-Gaussianity
  - `spectral_stationarity_test()`: Test for spectral stationarity

**Functions:**
- `runs_test()`: Convenience function for runs test
- `variance_ratio_test()`: Convenience function for variance test
- `mean_stationarity_test()`: Convenience function for mean test
- `kurtosis_test()`: Convenience function for kurtosis test

## Running Tests

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=nostalisa tests/
```

## Development

### Project Structure

```
NoStaLISA/
├── nostalisa/           # Main package
│   ├── __init__.py      # Package initialization
│   ├── injection.py     # Injection module
│   ├── fitting.py       # Fitting module
│   └── test.py          # Test module
├── tests/               # Test suite
│   ├── test_injection.py
│   ├── test_fitting.py
│   └── test_test.py
├── README.md            # This file
├── setup.py             # Setup script
├── pyproject.toml       # Project configuration
└── requirements.txt     # Dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use this package in your research, please cite:

```
@software{nostalisa,
  title = {NoStaLISA: Non-Stationary Noise Analysis for LISA},
  author = {LISA Collaboration},
  year = {2024},
  url = {https://github.com/lorenzsp/NoStaLISA}
}
```

## Contact

For questions and support, please open an issue on GitHub.