"""Tests for the fitting module."""

import numpy as np
import pytest
from nostalisa.fitting import (
    PSDEstimator,
    PSDFitter,
    welch_psd,
    periodogram_psd,
    fit_psd_model
)


class TestPSDEstimator:
    """Test cases for PSDEstimator class."""
    
    def test_initialization(self):
        """Test PSD estimator initialization."""
        estimator = PSDEstimator(sampling_frequency=10.0, method='welch')
        assert estimator.sampling_frequency == 10.0
        assert estimator.method == 'welch'
    
    def test_welch_method(self):
        """Test Welch's method for PSD estimation."""
        # Generate white noise
        np.random.seed(42)
        data = np.random.randn(1000)
        
        estimator = PSDEstimator(sampling_frequency=100.0, method='welch')
        freqs, psd = estimator.estimate(data)
        
        assert len(freqs) > 0
        assert len(psd) == len(freqs)
        assert np.all(freqs >= 0)
        assert np.all(psd >= 0)
    
    def test_periodogram_method(self):
        """Test periodogram method for PSD estimation."""
        # Generate white noise
        np.random.seed(42)
        data = np.random.randn(1000)
        
        estimator = PSDEstimator(sampling_frequency=100.0, method='periodogram')
        freqs, psd = estimator.estimate(data)
        
        assert len(freqs) > 0
        assert len(psd) == len(freqs)
        assert np.all(freqs >= 0)
        assert np.all(psd >= 0)
    
    def test_time_varying_psd(self):
        """Test time-varying PSD estimation (spectrogram)."""
        # Generate white noise
        np.random.seed(42)
        data = np.random.randn(1000)
        
        estimator = PSDEstimator(sampling_frequency=100.0, method='welch')
        freqs, times, Sxx = estimator.estimate_time_varying(data)
        
        assert len(freqs) > 0
        assert len(times) > 0
        assert Sxx.shape == (len(freqs), len(times))
        assert np.all(freqs >= 0)
        assert np.all(Sxx >= 0)
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        estimator = PSDEstimator(sampling_frequency=100.0, method='invalid')
        data = np.random.randn(100)
        
        with pytest.raises(ValueError):
            estimator.estimate(data)


class TestPSDFitter:
    """Test cases for PSDFitter class."""
    
    def test_initialization(self):
        """Test PSD fitter initialization."""
        fitter = PSDFitter()
        assert fitter.model_func is not None
    
    def test_power_law_fit(self):
        """Test power law fitting."""
        # Generate synthetic power law PSD
        freqs = np.logspace(-2, 2, 100)
        A, alpha = 1.0, -2.0
        psd_true = A * freqs ** alpha
        
        # Add some noise
        np.random.seed(42)
        psd = psd_true * (1 + 0.1 * np.random.randn(len(freqs)))
        
        fitter = PSDFitter()
        popt, pcov = fitter.fit(freqs, psd)
        
        # Check that we recovered approximately correct parameters
        assert len(popt) == 3  # A, alpha, f0
        assert np.abs(popt[1] - alpha) < 0.5  # alpha should be close to -2.0
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        freqs = np.logspace(-2, 2, 100)
        params = [1.0, -2.0, 1.0]  # A, alpha, f0
        
        fitter = PSDFitter()
        psd = fitter.evaluate_model(freqs, params)
        
        assert len(psd) == len(freqs)
        assert np.all(psd > 0)
    
    def test_fit_with_invalid_data(self):
        """Test that fitting with invalid data raises error."""
        freqs = np.array([0, 0, 0])
        psd = np.array([0, 0, 0])
        
        fitter = PSDFitter()
        with pytest.raises(ValueError):
            fitter.fit(freqs, psd)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_welch_psd_function(self):
        """Test welch_psd convenience function."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        freqs, psd = welch_psd(data, sampling_frequency=100.0)
        
        assert len(freqs) > 0
        assert len(psd) == len(freqs)
        assert np.all(freqs >= 0)
        assert np.all(psd >= 0)
    
    def test_periodogram_psd_function(self):
        """Test periodogram_psd convenience function."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        freqs, psd = periodogram_psd(data, sampling_frequency=100.0)
        
        assert len(freqs) > 0
        assert len(psd) == len(freqs)
        assert np.all(freqs >= 0)
        assert np.all(psd >= 0)
    
    def test_fit_psd_model_function(self):
        """Test fit_psd_model convenience function."""
        # Generate synthetic power law PSD
        freqs = np.logspace(-2, 2, 100)
        A, alpha = 1.0, -2.0
        psd_true = A * freqs ** alpha
        
        np.random.seed(42)
        psd = psd_true * (1 + 0.1 * np.random.randn(len(freqs)))
        
        popt, pcov = fit_psd_model(freqs, psd, model='power_law')
        
        assert len(popt) == 3  # A, alpha, f0
        assert pcov is not None
    
    def test_fit_psd_model_invalid_model(self):
        """Test that invalid model raises error."""
        freqs = np.logspace(-2, 2, 100)
        psd = np.ones_like(freqs)
        
        with pytest.raises(ValueError):
            fit_psd_model(freqs, psd, model='invalid_model')
