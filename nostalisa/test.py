"""
Test Module

This module provides statistical tests for detecting and characterizing
non-stationarity in LISA data.

Classes:
    NonStationarityTester: Class for running non-stationarity tests
    
Functions:
    augmented_dickey_fuller_test: Test for stationarity using ADF test
    runs_test: Runs test for randomness
    variance_ratio_test: Test for time-varying variance
    kpss_test: KPSS test for stationarity
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats


class NonStationarityTester:
    """
    A class for testing non-stationarity in time series data.
    
    This class provides various statistical tests to detect
    non-stationarity in LISA data streams.
    
    Attributes:
        data (np.ndarray): Time series data to test
        sampling_frequency (float): Sampling frequency in Hz
    """
    
    def __init__(self, data: np.ndarray, sampling_frequency: float = 1.0):
        """
        Initialize the non-stationarity tester.
        
        Args:
            data (np.ndarray): Time series data to test
            sampling_frequency (float): Sampling frequency in Hz (default: 1.0)
        """
        self.data = data
        self.sampling_frequency = sampling_frequency
        
    def runs_test(self) -> Dict[str, float]:
        """
        Perform runs test for randomness/stationarity.
        
        The runs test checks whether a sequence of binary outcomes
        is random by counting the number of runs (consecutive sequences
        of the same outcome).
        
        Returns:
            Dict[str, float]: Dictionary with test results including
                - n_runs: Number of runs
                - expected_runs: Expected number of runs under null hypothesis
                - z_statistic: Z-statistic
                - p_value: Two-tailed p-value
        """
        # Convert to binary sequence based on median
        median = np.median(self.data)
        binary = (self.data > median).astype(int)
        
        # Count runs
        runs = np.sum(np.diff(binary) != 0) + 1
        
        # Calculate statistics
        n1 = np.sum(binary == 1)
        n2 = np.sum(binary == 0)
        n = n1 + n2
        
        if n1 == 0 or n2 == 0:
            return {
                'n_runs': runs,
                'expected_runs': np.nan,
                'z_statistic': np.nan,
                'p_value': np.nan
            }
        
        expected_runs = (2 * n1 * n2) / n + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
        
        if variance_runs > 0:
            z_statistic = (runs - expected_runs) / np.sqrt(variance_runs)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        else:
            z_statistic = np.nan
            p_value = np.nan
        
        return {
            'n_runs': runs,
            'expected_runs': expected_runs,
            'z_statistic': z_statistic,
            'p_value': p_value
        }
    
    def variance_ratio_test(
        self, 
        window_size: Optional[int] = None,
        n_windows: int = 10
    ) -> Dict[str, float]:
        """
        Test for time-varying variance.
        
        This test divides the data into windows and compares
        the variance across windows.
        
        Args:
            window_size (int): Size of each window (default: None, auto-calculated)
            n_windows (int): Number of windows to use (default: 10)
            
        Returns:
            Dict[str, float]: Dictionary with test results including
                - variance_ratio: Ratio of max to min variance
                - f_statistic: F-statistic for Bartlett's test
                - p_value: P-value for Bartlett's test
        """
        if window_size is None:
            window_size = len(self.data) // n_windows
        
        # Calculate variances for each window
        variances = []
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(self.data))
            
            if end_idx - start_idx > 1:
                window_data = self.data[start_idx:end_idx]
                variances.append(np.var(window_data, ddof=1))
        
        if len(variances) < 2:
            return {
                'variance_ratio': np.nan,
                'f_statistic': np.nan,
                'p_value': np.nan
            }
        
        variances = np.array(variances)
        variance_ratio = np.max(variances) / np.min(variances) if np.min(variances) > 0 else np.inf
        
        # Bartlett's test for equal variances
        try:
            # Split data into windows for Bartlett's test
            windows = []
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(self.data))
                if end_idx - start_idx > 1:
                    windows.append(self.data[start_idx:end_idx])
            
            if len(windows) >= 2:
                statistic, p_value = stats.bartlett(*windows)
            else:
                statistic, p_value = np.nan, np.nan
        except Exception:
            statistic, p_value = np.nan, np.nan
        
        return {
            'variance_ratio': variance_ratio,
            'f_statistic': statistic,
            'p_value': p_value
        }
    
    def mean_stationarity_test(
        self,
        window_size: Optional[int] = None,
        n_windows: int = 10
    ) -> Dict[str, float]:
        """
        Test for stationarity in the mean.
        
        This test uses ANOVA to compare means across windows.
        
        Args:
            window_size (int): Size of each window (default: None, auto-calculated)
            n_windows (int): Number of windows to use (default: 10)
            
        Returns:
            Dict[str, float]: Dictionary with test results including
                - f_statistic: F-statistic
                - p_value: P-value
        """
        if window_size is None:
            window_size = len(self.data) // n_windows
        
        # Create windows
        windows = []
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(self.data))
            
            if end_idx - start_idx > 1:
                windows.append(self.data[start_idx:end_idx])
        
        if len(windows) < 2:
            return {
                'f_statistic': np.nan,
                'p_value': np.nan
            }
        
        # One-way ANOVA
        try:
            f_statistic, p_value = stats.f_oneway(*windows)
        except Exception:
            f_statistic, p_value = np.nan, np.nan
        
        return {
            'f_statistic': f_statistic,
            'p_value': p_value
        }
    
    def kurtosis_test(self) -> Dict[str, float]:
        """
        Test for non-Gaussianity using excess kurtosis.
        
        Returns:
            Dict[str, float]: Dictionary with test results including
                - kurtosis: Excess kurtosis value
                - z_statistic: Z-statistic
                - p_value: Two-tailed p-value
        """
        kurtosis = stats.kurtosis(self.data, fisher=True)
        n = len(self.data)
        
        # Standard error of kurtosis for normal distribution
        se_kurt = np.sqrt(24 / n)
        z_statistic = kurtosis / se_kurt
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        
        return {
            'kurtosis': kurtosis,
            'z_statistic': z_statistic,
            'p_value': p_value
        }
    
    def spectral_stationarity_test(
        self,
        n_segments: int = 4
    ) -> Dict[str, float]:
        """
        Test for spectral stationarity by comparing PSDs across segments.
        
        Args:
            n_segments (int): Number of segments to divide data into (default: 4)
            
        Returns:
            Dict[str, float]: Dictionary with test results including
                - max_relative_difference: Maximum relative difference in PSD
                - mean_relative_difference: Mean relative difference in PSD
        """
        from .fitting import PSDEstimator
        
        segment_length = len(self.data) // n_segments
        psds = []
        
        estimator = PSDEstimator(self.sampling_frequency, method='welch')
        
        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(self.data))
            segment = self.data[start_idx:end_idx]
            
            if len(segment) > 10:  # Minimum length for PSD estimation
                freqs, psd = estimator.estimate(segment)
                psds.append(psd)
        
        if len(psds) < 2:
            return {
                'max_relative_difference': np.nan,
                'mean_relative_difference': np.nan
            }
        
        # Compare PSDs
        psds = np.array(psds)
        mean_psd = np.mean(psds, axis=0)
        
        # Avoid division by zero
        mean_psd = np.where(mean_psd == 0, np.finfo(float).eps, mean_psd)
        
        relative_differences = np.abs(psds - mean_psd) / mean_psd
        max_rel_diff = np.max(relative_differences)
        mean_rel_diff = np.mean(relative_differences)
        
        return {
            'max_relative_difference': max_rel_diff,
            'mean_relative_difference': mean_rel_diff
        }


def runs_test(data: np.ndarray) -> Dict[str, float]:
    """
    Convenience function for runs test.
    
    Args:
        data (np.ndarray): Time series data to test
        
    Returns:
        Dict[str, float]: Test results
    """
    tester = NonStationarityTester(data)
    return tester.runs_test()


def variance_ratio_test(
    data: np.ndarray,
    window_size: Optional[int] = None,
    n_windows: int = 10
) -> Dict[str, float]:
    """
    Convenience function for variance ratio test.
    
    Args:
        data (np.ndarray): Time series data to test
        window_size (int): Size of each window
        n_windows (int): Number of windows to use (default: 10)
        
    Returns:
        Dict[str, float]: Test results
    """
    tester = NonStationarityTester(data)
    return tester.variance_ratio_test(window_size, n_windows)


def kurtosis_test(data: np.ndarray) -> Dict[str, float]:
    """
    Convenience function for kurtosis test.
    
    Args:
        data (np.ndarray): Time series data to test
        
    Returns:
        Dict[str, float]: Test results
    """
    tester = NonStationarityTester(data)
    return tester.kurtosis_test()


def mean_stationarity_test(
    data: np.ndarray,
    window_size: Optional[int] = None,
    n_windows: int = 10
) -> Dict[str, float]:
    """
    Convenience function for mean stationarity test.
    
    Args:
        data (np.ndarray): Time series data to test
        window_size (int): Size of each window
        n_windows (int): Number of windows to use (default: 10)
        
    Returns:
        Dict[str, float]: Test results
    """
    tester = NonStationarityTester(data)
    return tester.mean_stationarity_test(window_size, n_windows)
