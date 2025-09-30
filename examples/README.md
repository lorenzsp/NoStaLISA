# Examples

This directory contains example scripts demonstrating the NoStaLISA package.

## Running the Examples

### Demo Script

The `demo.py` script demonstrates all three main modules of NoStaLISA:

```bash
python examples/demo.py
```

This script will:
1. Generate synthetic time series data
2. Inject various non-stationary features (glitches, gaps, non-Gaussian noise)
3. Estimate PSDs using Welch's method
4. Run statistical tests for non-stationarity detection

## Creating Your Own Analysis

Use these examples as templates for your own LISA noise analysis:

```python
import numpy as np
from nostalisa.injection import NonStationaryNoiseInjector
from nostalisa.fitting import PSDEstimator
from nostalisa.test import NonStationarityTester

# Your analysis here
```
