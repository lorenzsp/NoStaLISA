"""
NoStaLISA: Non-Stationary Noise Analysis for LISA

A Python package for investigating non-stationary noise in LISA 
(Laser Interferometer Space Antenna).

Modules:
    - injection: Non-stationary noise injection tools
    - fitting: PSD estimation and fitting algorithms
    - test: Non-stationarity tests and statistical methods
"""

__version__ = "0.1.0"
__author__ = "LISA Collaboration"

from . import injection
from . import fitting
from . import test

__all__ = ["injection", "fitting", "test", "__version__"]
