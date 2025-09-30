"""
Fitting Module

This module provides tools for PSD (Power Spectral Density) estimation and fitting
for LISA data analysis.

Classes:
    PSDEstimator: Class for estimating power spectral densities
    PSDFitter: Class for fitting parametric models to PSDs
    
Functions:
    welch_psd: Compute PSD using Welch's method
    periodogram_psd: Compute PSD using periodogram method
    fit_psd_model: Fit a parametric model to estimated PSD
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict
from scipy import signal
from scipy.optimize import curve_fit


class PSDEstimator:
    """
    A class for estimating power spectral densities from time series data.
    
    This class provides various methods for PSD estimation including
    Welch's method and periodogram.
    
    Attributes:
        sampling_frequency (float): Sampling frequency in Hz
        method (str): Method for PSD estimation ('welch' or 'periodogram')
    """
    
    def __init__(self, sampling_frequency: float = 1.0, method: str = 'welch'):
        """
        Initialize the PSD estimator.
        
        Args:
            sampling_frequency (float): Sampling frequency in Hz (default: 1.0)
            method (str): Method for PSD estimation (default: 'welch')
        """
        self.sampling_frequency = sampling_frequency
        self.method = method
        
    def estimate(
        self, 
        data: np.ndarray,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        window: str = 'hann',
        detrend: str = 'constant',
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the PSD of the input data.
        
        Args:
            data (np.ndarray): Input time series data
            nperseg (int): Length of each segment (default: 256 for Welch's method)
            noverlap (int): Number of points to overlap between segments (default: nperseg // 2)
            window (str): Window function to use (default: 'hann')
            detrend (str): Detrending method (default: 'constant')
            **kwargs: Additional arguments passed to the PSD estimation function
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Frequency array and PSD array
        """
        if self.method == 'welch':
            return self._welch(data, nperseg, noverlap, window, detrend, **kwargs)
        elif self.method == 'periodogram':
            return self._periodogram(data, window, detrend, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _welch(
        self,
        data: np.ndarray,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        window: str = 'hann',
        detrend: str = 'constant',
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PSD using Welch's method.
        
        Args:
            data (np.ndarray): Input time series data
            nperseg (int): Length of each segment
            noverlap (int): Number of points to overlap between segments
            window (str): Window function to use
            detrend (str): Detrending method
            **kwargs: Additional arguments
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Frequency array and PSD array
        """
        if nperseg is None:
            nperseg = min(256, len(data))
        if noverlap is None:
            noverlap = nperseg // 2
            
        freqs, psd = signal.welch(
            data,
            fs=self.sampling_frequency,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            **kwargs
        )
        return freqs, psd
    
    def _periodogram(
        self,
        data: np.ndarray,
        window: str = 'hann',
        detrend: str = 'constant',
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PSD using the periodogram method.
        
        Args:
            data (np.ndarray): Input time series data
            window (str): Window function to use
            detrend (str): Detrending method
            **kwargs: Additional arguments
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Frequency array and PSD array
        """
        freqs, psd = signal.periodogram(
            data,
            fs=self.sampling_frequency,
            window=window,
            detrend=detrend,
            **kwargs
        )
        return freqs, psd
    
    def estimate_time_varying(
        self,
        data: np.ndarray,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        window: str = 'hann',
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate time-varying PSD using the spectrogram.
        
        Args:
            data (np.ndarray): Input time series data
            nperseg (int): Length of each segment
            noverlap (int): Number of points to overlap between segments
            window (str): Window function to use
            **kwargs: Additional arguments
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequency array, time array, and spectrogram
        """
        if nperseg is None:
            nperseg = min(256, len(data))
        if noverlap is None:
            noverlap = nperseg // 2
            
        freqs, times, Sxx = signal.spectrogram(
            data,
            fs=self.sampling_frequency,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            **kwargs
        )
        return freqs, times, Sxx


class PSDFitter:
    """
    A class for fitting parametric models to estimated PSDs.
    
    This class provides methods to fit various parametric models
    to power spectral densities.
    
    Attributes:
        model_func (Callable): Model function to fit
    """
    
    def __init__(self, model_func: Optional[Callable] = None):
        """
        Initialize the PSD fitter.
        
        Args:
            model_func (Callable): Model function to fit (default: None, uses power law)
        """
        if model_func is None:
            self.model_func = self._power_law_model
        else:
            self.model_func = model_func
    
    @staticmethod
    def _power_law_model(f: np.ndarray, A: float, alpha: float, f0: float = 1.0) -> np.ndarray:
        """
        Power law model: PSD(f) = A * (f/f0)^alpha
        
        Args:
            f (np.ndarray): Frequency array
            A (float): Amplitude
            alpha (float): Spectral index
            f0 (float): Reference frequency (default: 1.0)
            
        Returns:
            np.ndarray: Model PSD values
        """
        return A * (f / f0) ** alpha
    
    @staticmethod
    def _broken_power_law_model(
        f: np.ndarray, 
        A: float, 
        alpha1: float, 
        alpha2: float, 
        f_break: float
    ) -> np.ndarray:
        """
        Broken power law model.
        
        Args:
            f (np.ndarray): Frequency array
            A (float): Amplitude
            alpha1 (float): Spectral index below break
            alpha2 (float): Spectral index above break
            f_break (float): Break frequency
            
        Returns:
            np.ndarray: Model PSD values
        """
        psd = np.zeros_like(f)
        mask_low = f < f_break
        mask_high = f >= f_break
        
        psd[mask_low] = A * (f[mask_low] / f_break) ** alpha1
        psd[mask_high] = A * (f[mask_high] / f_break) ** alpha2
        
        return psd
    
    def fit(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        p0: Optional[list] = None,
        bounds: Tuple = (-np.inf, np.inf),
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the model to the PSD data.
        
        Args:
            freqs (np.ndarray): Frequency array
            psd (np.ndarray): PSD array
            p0 (list): Initial parameter guesses (default: None)
            bounds (Tuple): Bounds for parameters (default: (-inf, inf))
            **kwargs: Additional arguments for curve_fit
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Optimal parameters and covariance matrix
        """
        # Filter out zero or negative frequencies and PSDs
        valid_mask = (freqs > 0) & (psd > 0)
        freqs_valid = freqs[valid_mask]
        psd_valid = psd[valid_mask]
        
        if len(freqs_valid) == 0:
            raise ValueError("No valid frequency-PSD pairs for fitting")
        
        try:
            popt, pcov = curve_fit(
                self.model_func,
                freqs_valid,
                psd_valid,
                p0=p0,
                bounds=bounds,
                **kwargs
            )
            return popt, pcov
        except RuntimeError as e:
            raise RuntimeError(f"Fitting failed: {e}")
    
    def evaluate_model(self, freqs: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the model at given frequencies with given parameters.
        
        Args:
            freqs (np.ndarray): Frequency array
            params (np.ndarray): Model parameters
            
        Returns:
            np.ndarray: Model PSD values
        """
        return self.model_func(freqs, *params)


def welch_psd(
    data: np.ndarray,
    sampling_frequency: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute PSD using Welch's method.
    
    Args:
        data (np.ndarray): Input time series data
        sampling_frequency (float): Sampling frequency in Hz (default: 1.0)
        nperseg (int): Length of each segment (default: 256)
        noverlap (int): Number of points to overlap between segments
        window (str): Window function to use (default: 'hann')
        **kwargs: Additional arguments
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequency array and PSD array
    """
    estimator = PSDEstimator(sampling_frequency, method='welch')
    return estimator.estimate(data, nperseg, noverlap, window, **kwargs)


def periodogram_psd(
    data: np.ndarray,
    sampling_frequency: float = 1.0,
    window: str = 'hann',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute PSD using the periodogram method.
    
    Args:
        data (np.ndarray): Input time series data
        sampling_frequency (float): Sampling frequency in Hz (default: 1.0)
        window (str): Window function to use (default: 'hann')
        **kwargs: Additional arguments
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequency array and PSD array
    """
    estimator = PSDEstimator(sampling_frequency, method='periodogram')
    return estimator.estimate(data, window=window, **kwargs)


def fit_psd_model(
    freqs: np.ndarray,
    psd: np.ndarray,
    model: str = 'power_law',
    p0: Optional[list] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to fit a parametric model to PSD data.
    
    Args:
        freqs (np.ndarray): Frequency array
        psd (np.ndarray): PSD array
        model (str): Model type ('power_law' or 'broken_power_law') (default: 'power_law')
        p0 (list): Initial parameter guesses
        **kwargs: Additional arguments
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Optimal parameters and covariance matrix
    """
    if model == 'power_law':
        fitter = PSDFitter(PSDFitter._power_law_model)
    elif model == 'broken_power_law':
        fitter = PSDFitter(PSDFitter._broken_power_law_model)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return fitter.fit(freqs, psd, p0, **kwargs)
