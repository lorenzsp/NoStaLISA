"""
Injection Module

This module provides tools for injecting non-stationary noise into LISA data streams.

Classes:
    NonStationaryNoiseInjector: Main class for noise injection
    
Functions:
    inject_glitch: Inject transient glitches into the data
    inject_gap: Inject data gaps
    inject_non_gaussian: Inject non-Gaussian noise features
"""

import numpy as np
from typing import Optional, Union, Callable


class NonStationaryNoiseInjector:
    """
    A class for injecting non-stationary noise features into LISA data.
    
    This class provides methods to inject various types of non-stationary
    features including glitches, gaps, and non-Gaussian noise.
    
    Attributes:
        sampling_frequency (float): Sampling frequency in Hz
        duration (float): Duration of the data segment in seconds
    """
    
    def __init__(self, sampling_frequency: float = 1.0, duration: float = 1.0):
        """
        Initialize the noise injector.
        
        Args:
            sampling_frequency (float): Sampling frequency in Hz (default: 1.0)
            duration (float): Duration of the data segment in seconds (default: 1.0)
        """
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.n_samples = int(sampling_frequency * duration)
        
    def inject_glitch(
        self, 
        data: np.ndarray, 
        time: float,
        amplitude: float = 1.0,
        width: float = 0.1,
        form: str = 'gaussian'
    ) -> np.ndarray:
        """
        Inject a transient glitch into the data.
        
        Args:
            data (np.ndarray): Input data array
            time (float): Time at which to inject the glitch (in seconds)
            amplitude (float): Amplitude of the glitch (default: 1.0)
            width (float): Width of the glitch in seconds (default: 0.1)
            form (str): Form of the glitch - 'gaussian', 'sine-gaussian', or 'delta' (default: 'gaussian')
            
        Returns:
            np.ndarray: Data with injected glitch
        """
        data_with_glitch = data.copy()
        t_array = np.arange(len(data)) / self.sampling_frequency
        t_center = time
        
        if form == 'gaussian':
            glitch = amplitude * np.exp(-0.5 * ((t_array - t_center) / width) ** 2)
        elif form == 'sine-gaussian':
            f_central = 1.0 / width  # Central frequency
            glitch = amplitude * np.sin(2 * np.pi * f_central * (t_array - t_center)) * \
                     np.exp(-0.5 * ((t_array - t_center) / width) ** 2)
        elif form == 'delta':
            glitch = np.zeros_like(t_array)
            idx = int(time * self.sampling_frequency)
            if 0 <= idx < len(glitch):
                glitch[idx] = amplitude
        else:
            raise ValueError(f"Unknown glitch form: {form}")
            
        return data_with_glitch + glitch
    
    def inject_gap(
        self, 
        data: np.ndarray, 
        start_time: float,
        gap_duration: float,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Inject a data gap.
        
        Args:
            data (np.ndarray): Input data array
            start_time (float): Start time of the gap in seconds
            gap_duration (float): Duration of the gap in seconds
            fill_value (float): Value to fill the gap with (default: 0.0)
            
        Returns:
            np.ndarray: Data with injected gap
        """
        data_with_gap = data.copy()
        start_idx = int(start_time * self.sampling_frequency)
        end_idx = int((start_time + gap_duration) * self.sampling_frequency)
        
        start_idx = max(0, start_idx)
        end_idx = min(len(data), end_idx)
        
        data_with_gap[start_idx:end_idx] = fill_value
        return data_with_gap
    
    def inject_non_gaussian(
        self, 
        data: np.ndarray,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        distribution: str = 'lognormal',
        **kwargs
    ) -> np.ndarray:
        """
        Inject non-Gaussian noise features.
        
        Args:
            data (np.ndarray): Input data array
            start_time (float): Start time for non-Gaussian injection (default: 0.0)
            end_time (float): End time for non-Gaussian injection (default: None, uses full duration)
            distribution (str): Type of distribution - 'lognormal', 'exponential', or 'cauchy' (default: 'lognormal')
            **kwargs: Additional parameters for the distribution
            
        Returns:
            np.ndarray: Data with injected non-Gaussian noise
        """
        data_with_noise = data.copy()
        
        if end_time is None:
            end_time = self.duration
            
        start_idx = int(start_time * self.sampling_frequency)
        end_idx = int(end_time * self.sampling_frequency)
        n_points = end_idx - start_idx
        
        if distribution == 'lognormal':
            mean = kwargs.get('mean', 0.0)
            sigma = kwargs.get('sigma', 1.0)
            noise = np.random.lognormal(mean, sigma, n_points)
        elif distribution == 'exponential':
            scale = kwargs.get('scale', 1.0)
            noise = np.random.exponential(scale, n_points)
        elif distribution == 'cauchy':
            loc = kwargs.get('loc', 0.0)
            scale = kwargs.get('scale', 1.0)
            noise = np.random.standard_cauchy(n_points) * scale + loc
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        data_with_noise[start_idx:end_idx] += noise
        return data_with_noise


def inject_glitch(
    data: np.ndarray,
    sampling_frequency: float,
    time: float,
    amplitude: float = 1.0,
    width: float = 0.1,
    form: str = 'gaussian'
) -> np.ndarray:
    """
    Convenience function to inject a glitch into data.
    
    Args:
        data (np.ndarray): Input data array
        sampling_frequency (float): Sampling frequency in Hz
        time (float): Time at which to inject the glitch (in seconds)
        amplitude (float): Amplitude of the glitch (default: 1.0)
        width (float): Width of the glitch in seconds (default: 0.1)
        form (str): Form of the glitch - 'gaussian', 'sine-gaussian', or 'delta' (default: 'gaussian')
        
    Returns:
        np.ndarray: Data with injected glitch
    """
    duration = len(data) / sampling_frequency
    injector = NonStationaryNoiseInjector(sampling_frequency, duration)
    return injector.inject_glitch(data, time, amplitude, width, form)


def inject_gap(
    data: np.ndarray,
    sampling_frequency: float,
    start_time: float,
    gap_duration: float,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Convenience function to inject a data gap.
    
    Args:
        data (np.ndarray): Input data array
        sampling_frequency (float): Sampling frequency in Hz
        start_time (float): Start time of the gap in seconds
        gap_duration (float): Duration of the gap in seconds
        fill_value (float): Value to fill the gap with (default: 0.0)
        
    Returns:
        np.ndarray: Data with injected gap
    """
    duration = len(data) / sampling_frequency
    injector = NonStationaryNoiseInjector(sampling_frequency, duration)
    return injector.inject_gap(data, start_time, gap_duration, fill_value)


def inject_non_gaussian(
    data: np.ndarray,
    sampling_frequency: float,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    distribution: str = 'lognormal',
    **kwargs
) -> np.ndarray:
    """
    Convenience function to inject non-Gaussian noise.
    
    Args:
        data (np.ndarray): Input data array
        sampling_frequency (float): Sampling frequency in Hz
        start_time (float): Start time for non-Gaussian injection (default: 0.0)
        end_time (float): End time for non-Gaussian injection (default: None, uses full duration)
        distribution (str): Type of distribution - 'lognormal', 'exponential', or 'cauchy' (default: 'lognormal')
        **kwargs: Additional parameters for the distribution
        
    Returns:
        np.ndarray: Data with injected non-Gaussian noise
    """
    duration = len(data) / sampling_frequency
    injector = NonStationaryNoiseInjector(sampling_frequency, duration)
    return injector.inject_non_gaussian(data, start_time, end_time, distribution, **kwargs)
