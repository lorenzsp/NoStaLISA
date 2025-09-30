"""Tests for the test module."""

import numpy as np
import pytest
from nostalisa.test import (
    NonStationarityTester,
    runs_test,
    variance_ratio_test,
    kurtosis_test,
    mean_stationarity_test
)


class TestNonStationarityTester:
    """Test cases for NonStationarityTester class."""
    
    def test_initialization(self):
        """Test tester initialization."""
        data = np.random.randn(100)
        tester = NonStationarityTester(data, sampling_frequency=10.0)
        assert len(tester.data) == 100
        assert tester.sampling_frequency == 10.0
    
    def test_runs_test(self):
        """Test runs test for randomness."""
        # Generate random data
        np.random.seed(42)
        data = np.random.randn(100)
        
        tester = NonStationarityTester(data)
        results = tester.runs_test()
        
        assert 'n_runs' in results
        assert 'expected_runs' in results
        assert 'z_statistic' in results
        assert 'p_value' in results
        
        # For random data, p-value should not indicate non-randomness (typically > 0.05)
        assert results['p_value'] > 0.01
    
    def test_runs_test_non_random(self):
        """Test runs test on clearly non-random data."""
        # Generate alternating data (very non-random)
        data = np.array([1, -1, 1, -1, 1, -1, 1, -1] * 10)
        
        tester = NonStationarityTester(data)
        results = tester.runs_test()
        
        # Should detect non-randomness
        assert results['n_runs'] > results['expected_runs']
    
    def test_variance_ratio_test(self):
        """Test variance ratio test."""
        # Generate data with constant variance
        np.random.seed(42)
        data = np.random.randn(1000)
        
        tester = NonStationarityTester(data)
        results = tester.variance_ratio_test(n_windows=10)
        
        assert 'variance_ratio' in results
        assert 'f_statistic' in results
        assert 'p_value' in results
        
        # Variance ratio should be close to 1 for stationary data
        assert results['variance_ratio'] < 5.0
    
    def test_variance_ratio_test_non_stationary(self):
        """Test variance ratio test on non-stationary data."""
        # Generate data with increasing variance
        np.random.seed(42)
        t = np.arange(1000)
        data = np.random.randn(1000) * (1 + t / 1000)
        
        tester = NonStationarityTester(data)
        results = tester.variance_ratio_test(n_windows=10)
        
        # Should detect variance change
        assert results['variance_ratio'] > 1.5
        # Small p-value indicates rejection of equal variances
        assert results['p_value'] < 0.1
    
    def test_mean_stationarity_test(self):
        """Test mean stationarity test."""
        # Generate stationary data (constant mean)
        np.random.seed(42)
        data = np.random.randn(1000)
        
        tester = NonStationarityTester(data)
        results = tester.mean_stationarity_test(n_windows=10)
        
        assert 'f_statistic' in results
        assert 'p_value' in results
        
        # For stationary data, should not reject null hypothesis
        assert results['p_value'] > 0.01
    
    def test_mean_stationarity_test_non_stationary(self):
        """Test mean stationarity test on non-stationary data."""
        # Generate data with changing mean
        np.random.seed(42)
        data = np.concatenate([
            np.random.randn(500),
            np.random.randn(500) + 5.0  # Shift mean
        ])
        
        tester = NonStationarityTester(data)
        results = tester.mean_stationarity_test(n_windows=10)
        
        # Should detect mean change
        assert results['p_value'] < 0.01
    
    def test_kurtosis_test(self):
        """Test kurtosis test for Gaussianity."""
        # Generate Gaussian data
        np.random.seed(42)
        data = np.random.randn(1000)
        
        tester = NonStationarityTester(data)
        results = tester.kurtosis_test()
        
        assert 'kurtosis' in results
        assert 'z_statistic' in results
        assert 'p_value' in results
        
        # Kurtosis should be close to 0 for Gaussian data
        assert abs(results['kurtosis']) < 1.0
    
    def test_kurtosis_test_non_gaussian(self):
        """Test kurtosis test on non-Gaussian data."""
        # Generate uniform data (has negative excess kurtosis)
        np.random.seed(42)
        data = np.random.uniform(-1, 1, 1000)
        
        tester = NonStationarityTester(data)
        results = tester.kurtosis_test()
        
        # Uniform distribution has negative excess kurtosis
        assert results['kurtosis'] < -0.5
    
    def test_spectral_stationarity_test(self):
        """Test spectral stationarity test."""
        # Generate stationary data
        np.random.seed(42)
        data = np.random.randn(1000)
        
        tester = NonStationarityTester(data, sampling_frequency=100.0)
        results = tester.spectral_stationarity_test(n_segments=4)
        
        assert 'max_relative_difference' in results
        assert 'mean_relative_difference' in results
        
        # For stationary data, differences should be relatively small
        assert results['mean_relative_difference'] < 2.0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_runs_test_function(self):
        """Test runs_test convenience function."""
        np.random.seed(42)
        data = np.random.randn(100)
        
        results = runs_test(data)
        
        assert 'n_runs' in results
        assert 'p_value' in results
    
    def test_variance_ratio_test_function(self):
        """Test variance_ratio_test convenience function."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        results = variance_ratio_test(data, n_windows=10)
        
        assert 'variance_ratio' in results
        assert 'p_value' in results
    
    def test_kurtosis_test_function(self):
        """Test kurtosis_test convenience function."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        results = kurtosis_test(data)
        
        assert 'kurtosis' in results
        assert 'p_value' in results
    
    def test_mean_stationarity_test_function(self):
        """Test mean_stationarity_test convenience function."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        results = mean_stationarity_test(data, n_windows=10)
        
        assert 'f_statistic' in results
        assert 'p_value' in results
