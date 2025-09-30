"""Tests for the injection module."""

import numpy as np
import pytest
from nostalisa.injection import (
    NonStationaryNoiseInjector,
    inject_glitch,
    inject_gap,
    inject_non_gaussian
)


class TestNonStationaryNoiseInjector:
    """Test cases for NonStationaryNoiseInjector class."""
    
    def test_initialization(self):
        """Test injector initialization."""
        injector = NonStationaryNoiseInjector(sampling_frequency=10.0, duration=1.0)
        assert injector.sampling_frequency == 10.0
        assert injector.duration == 1.0
        assert injector.n_samples == 10
    
    def test_inject_glitch_gaussian(self):
        """Test Gaussian glitch injection."""
        injector = NonStationaryNoiseInjector(sampling_frequency=100.0, duration=1.0)
        data = np.zeros(100)
        
        data_with_glitch = injector.inject_glitch(
            data, 
            time=0.5, 
            amplitude=1.0, 
            width=0.1, 
            form='gaussian'
        )
        
        # Check that glitch was added
        assert not np.allclose(data, data_with_glitch)
        assert len(data_with_glitch) == len(data)
        # Check that the maximum is near the injection time
        max_idx = np.argmax(np.abs(data_with_glitch))
        assert abs(max_idx - 50) < 10  # Within 10 samples of expected location
    
    def test_inject_glitch_sine_gaussian(self):
        """Test sine-Gaussian glitch injection."""
        injector = NonStationaryNoiseInjector(sampling_frequency=100.0, duration=1.0)
        data = np.zeros(100)
        
        data_with_glitch = injector.inject_glitch(
            data, 
            time=0.5, 
            amplitude=1.0, 
            width=0.1, 
            form='sine-gaussian'
        )
        
        assert not np.allclose(data, data_with_glitch)
        assert len(data_with_glitch) == len(data)
    
    def test_inject_glitch_delta(self):
        """Test delta glitch injection."""
        injector = NonStationaryNoiseInjector(sampling_frequency=100.0, duration=1.0)
        data = np.zeros(100)
        
        data_with_glitch = injector.inject_glitch(
            data, 
            time=0.5, 
            amplitude=5.0, 
            form='delta'
        )
        
        # Check that exactly one sample is non-zero
        assert np.sum(data_with_glitch != 0) == 1
        assert np.max(np.abs(data_with_glitch)) == 5.0
    
    def test_inject_gap(self):
        """Test gap injection."""
        injector = NonStationaryNoiseInjector(sampling_frequency=100.0, duration=1.0)
        data = np.ones(100)
        
        data_with_gap = injector.inject_gap(
            data, 
            start_time=0.3, 
            gap_duration=0.2, 
            fill_value=0.0
        )
        
        # Check that gap was created
        assert np.sum(data_with_gap == 0.0) > 0
        # Gap should be approximately 20 samples (0.2 * 100)
        assert 15 <= np.sum(data_with_gap == 0.0) <= 25
    
    def test_inject_non_gaussian_lognormal(self):
        """Test lognormal non-Gaussian injection."""
        injector = NonStationaryNoiseInjector(sampling_frequency=100.0, duration=1.0)
        data = np.zeros(100)
        
        data_with_noise = injector.inject_non_gaussian(
            data,
            start_time=0.0,
            end_time=1.0,
            distribution='lognormal',
            mean=0.0,
            sigma=1.0
        )
        
        assert not np.allclose(data, data_with_noise)
        assert len(data_with_noise) == len(data)
        # All values should be >= original (lognormal is always positive)
        assert np.all(data_with_noise >= data)
    
    def test_inject_non_gaussian_exponential(self):
        """Test exponential non-Gaussian injection."""
        injector = NonStationaryNoiseInjector(sampling_frequency=100.0, duration=1.0)
        data = np.zeros(100)
        
        data_with_noise = injector.inject_non_gaussian(
            data,
            distribution='exponential',
            scale=1.0
        )
        
        assert not np.allclose(data, data_with_noise)
        assert len(data_with_noise) == len(data)
    
    def test_inject_non_gaussian_cauchy(self):
        """Test Cauchy non-Gaussian injection."""
        injector = NonStationaryNoiseInjector(sampling_frequency=100.0, duration=1.0)
        data = np.zeros(100)
        
        data_with_noise = injector.inject_non_gaussian(
            data,
            distribution='cauchy',
            loc=0.0,
            scale=1.0
        )
        
        assert not np.allclose(data, data_with_noise)
        assert len(data_with_noise) == len(data)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_inject_glitch_function(self):
        """Test inject_glitch convenience function."""
        data = np.zeros(100)
        data_with_glitch = inject_glitch(
            data,
            sampling_frequency=100.0,
            time=0.5,
            amplitude=1.0,
            width=0.1
        )
        
        assert not np.allclose(data, data_with_glitch)
        assert len(data_with_glitch) == len(data)
    
    def test_inject_gap_function(self):
        """Test inject_gap convenience function."""
        data = np.ones(100)
        data_with_gap = inject_gap(
            data,
            sampling_frequency=100.0,
            start_time=0.3,
            gap_duration=0.2
        )
        
        assert np.sum(data_with_gap == 0.0) > 0
    
    def test_inject_non_gaussian_function(self):
        """Test inject_non_gaussian convenience function."""
        data = np.zeros(100)
        data_with_noise = inject_non_gaussian(
            data,
            sampling_frequency=100.0,
            distribution='lognormal'
        )
        
        assert not np.allclose(data, data_with_noise)
        assert len(data_with_noise) == len(data)
