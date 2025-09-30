"""
Example script demonstrating NoStaLISA package functionality.

This script shows how to use the injection, fitting, and test modules
to analyze non-stationary noise in time series data.
"""

import numpy as np
from nostalisa.injection import NonStationaryNoiseInjector
from nostalisa.fitting import PSDEstimator, PSDFitter
from nostalisa.test import NonStationarityTester


def main():
    """Run example analyses."""
    print("=" * 70)
    print("NoStaLISA Example: Non-Stationary Noise Analysis for LISA")
    print("=" * 70)
    
    # Set parameters
    sampling_frequency = 100.0  # Hz
    duration = 10.0  # seconds
    n_samples = int(sampling_frequency * duration)
    
    # Generate clean Gaussian noise
    np.random.seed(42)
    clean_data = np.random.randn(n_samples)
    
    print(f"\nGenerated {n_samples} samples at {sampling_frequency} Hz")
    
    # ======================================================================
    # 1. INJECTION MODULE EXAMPLE
    # ======================================================================
    print("\n" + "=" * 70)
    print("1. INJECTION MODULE: Adding non-stationary features")
    print("=" * 70)
    
    injector = NonStationaryNoiseInjector(sampling_frequency, duration)
    
    # Inject a glitch
    data_with_glitch = injector.inject_glitch(
        clean_data.copy(),
        time=5.0,
        amplitude=10.0,
        width=0.1,
        form='gaussian'
    )
    print("\n  ✓ Injected Gaussian glitch at t=5.0s with amplitude=10.0")
    
    # Inject a gap
    data_with_gap = injector.inject_gap(
        clean_data.copy(),
        start_time=3.0,
        gap_duration=1.0,
        fill_value=0.0
    )
    n_zeros = np.sum(data_with_gap == 0.0)
    print(f"  ✓ Injected data gap from t=3.0s to t=4.0s ({n_zeros} samples)")
    
    # Inject non-Gaussian noise
    data_with_ng = injector.inject_non_gaussian(
        clean_data.copy(),
        distribution='lognormal',
        mean=0.0,
        sigma=0.5
    )
    print("  ✓ Injected lognormal non-Gaussian noise")
    
    # ======================================================================
    # 2. FITTING MODULE EXAMPLE
    # ======================================================================
    print("\n" + "=" * 70)
    print("2. FITTING MODULE: PSD estimation and fitting")
    print("=" * 70)
    
    estimator = PSDEstimator(sampling_frequency, method='welch')
    
    # Estimate PSD of clean data
    freqs_clean, psd_clean = estimator.estimate(clean_data, nperseg=256)
    print(f"\n  ✓ Estimated PSD using Welch's method ({len(freqs_clean)} frequency bins)")
    
    # Estimate PSD of data with glitch
    freqs_glitch, psd_glitch = estimator.estimate(data_with_glitch, nperseg=256)
    
    # Calculate ratio of PSDs
    psd_ratio = np.mean(psd_glitch) / np.mean(psd_clean)
    print(f"  ✓ PSD ratio (glitch/clean): {psd_ratio:.3f}")
    
    # Fit power law model
    fitter = PSDFitter()
    try:
        params, _ = fitter.fit(freqs_clean[1:], psd_clean[1:])  # Skip DC component
        print(f"  ✓ Fitted power law model: A={params[0]:.3e}, α={params[1]:.3f}")
    except Exception as e:
        print(f"  ⚠ Fitting failed (expected for white noise): {e}")
    
    # Estimate time-varying PSD
    freqs_tv, times_tv, spectrogram = estimator.estimate_time_varying(
        data_with_glitch, 
        nperseg=128
    )
    print(f"  ✓ Computed spectrogram: {spectrogram.shape[0]} freq × {spectrogram.shape[1]} time bins")
    
    # ======================================================================
    # 3. TEST MODULE EXAMPLE
    # ======================================================================
    print("\n" + "=" * 70)
    print("3. TEST MODULE: Non-stationarity detection")
    print("=" * 70)
    
    # Test clean data
    print("\n  Clean Gaussian data:")
    tester_clean = NonStationarityTester(clean_data, sampling_frequency)
    
    runs_result = tester_clean.runs_test()
    print(f"    - Runs test p-value: {runs_result['p_value']:.4f}")
    
    var_result = tester_clean.variance_ratio_test(n_windows=10)
    print(f"    - Variance ratio: {var_result['variance_ratio']:.3f} (p={var_result['p_value']:.4f})")
    
    mean_result = tester_clean.mean_stationarity_test(n_windows=10)
    print(f"    - Mean stationarity p-value: {mean_result['p_value']:.4f}")
    
    kurt_result = tester_clean.kurtosis_test()
    print(f"    - Excess kurtosis: {kurt_result['kurtosis']:.3f} (p={kurt_result['p_value']:.4f})")
    
    # Test data with glitch
    print("\n  Data with glitch:")
    tester_glitch = NonStationarityTester(data_with_glitch, sampling_frequency)
    
    runs_result_g = tester_glitch.runs_test()
    print(f"    - Runs test p-value: {runs_result_g['p_value']:.4f}")
    
    var_result_g = tester_glitch.variance_ratio_test(n_windows=10)
    print(f"    - Variance ratio: {var_result_g['variance_ratio']:.3f} (p={var_result_g['p_value']:.4f})")
    
    mean_result_g = tester_glitch.mean_stationarity_test(n_windows=10)
    print(f"    - Mean stationarity p-value: {mean_result_g['p_value']:.4f}")
    
    kurt_result_g = tester_glitch.kurtosis_test()
    print(f"    - Excess kurtosis: {kurt_result_g['kurtosis']:.3f} (p={kurt_result_g['p_value']:.4f})")
    
    spec_result_g = tester_glitch.spectral_stationarity_test(n_segments=4)
    print(f"    - Spectral stationarity max diff: {spec_result_g['max_relative_difference']:.3f}")
    
    # Test data with gap
    print("\n  Data with gap:")
    tester_gap = NonStationarityTester(data_with_gap, sampling_frequency)
    
    var_result_gap = tester_gap.variance_ratio_test(n_windows=10)
    print(f"    - Variance ratio: {var_result_gap['variance_ratio']:.3f} (p={var_result_gap['p_value']:.4f})")
    
    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nNoStaLISA successfully demonstrated:")
    print("  ✓ Injection of glitches, gaps, and non-Gaussian features")
    print("  ✓ PSD estimation using Welch's method and spectrogram")
    print("  ✓ Detection of non-stationarity using statistical tests")
    print("\nThe package is ready for analyzing non-stationary noise in LISA data!")
    print("=" * 70)


if __name__ == "__main__":
    main()
