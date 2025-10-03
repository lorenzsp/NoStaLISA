import numpy as np
import matplotlib.pyplot as plt 

def h(a, f0, fdot0, t, w_t=1.0):
    """
    Generate gravitational wave signal with chirping frequency.
    
    Parameters:
    - a: amplitude
    - f0: initial frequency [Hz]
    - fdot0: frequency derivative [Hz/s]
    - t: time array [s]
    - w_t: modulation window function (default=1.0)
    
    Returns:
    - Waveform time series
    """
    return w_t * 3e-22 * (a * np.sin((2*np.pi)*t*(f0 + fdot0 * t)))


def modulation(A, B, OMEGA, t, T_obs):
    """
    Generate cosine amplitude modulation.
    
    Parameters:
    - A: baseline amplitude
    - B: modulation depth
    - t: time array [s]
    - T_obs: observation time [s]
    
    Returns:
    - Modulation function w(t) = A + B*cos(2πt/T_obs)
    """
    return A + B * np.cos(2 * np.pi * OMEGA * t / T_obs)


def noise_PSD(f, TDI='TDI1'):
    """
    LISA noise power spectral density for different TDI channels.
    
    Parameters:
    - f: frequency array [Hz]
    - TDI: 'TDI1' or 'TDI2'
    
    Returns:
    - Power spectral density [strain²/Hz]
    """
    L = 2.5e9  # LISA arm length [m]
    c = 299792458  # speed of light [m/s]
    x = 2*np.pi*(L/c)*f
    
    # Acceleration noise
    Spm = (3e-15)**2 * (1 + ((4e-4)/f)**2) * (1 + (f/(8e-3))**4) * \
          (1/(2*np.pi*f))**4 * (2*np.pi*f/c)**2
    
    # Optical metrology noise
    Sop = (15e-12)**2 * (1 + ((2e-3)/f)**4) * ((2*np.pi*f)/c)**2
    
    S_val = (2*Spm*(3 + 2*np.cos(x) + np.cos(2*x)) + Sop*(2 + np.cos(x)))
    
    if TDI == 'TDI1':
        # Red noise at low frequencies
        S = 8*(np.sin(x)**2) * S_val
    elif TDI == 'TDI2':
        # White noise at low frequencies, stronger violet noise
        S = 32*np.sin(x)**2 * np.sin(2*x)**2 * S_val
    else:
        raise ValueError("TDI must be 'TDI1' or 'TDI2'")
    
    return S


def zero_pad(data):
    """
    Zero-pad data to next power of 2 for efficient FFT.
    
    Parameters:
    - data: input array
    
    Returns:
    - Zero-padded array with length 2^n
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), 'constant')


print("✓ Signal and noise functions defined")


@njit(parallel=True) 
def compute_covariance_slow(Cov, w_fft, PSD, Delta_f):
    """
    Compute N×N Fourier-domain covariance matrix for windowed process (slow version).
    
    Uses nested loops with Numba JIT compilation for verification purposes.
    
    Parameters:
    - Cov: preallocated N×N array
    - w_fft: FFT of window function (two-sided, fftshifted)
    - PSD: power spectral density (two-sided, fftshifted)
    - Delta_f: frequency resolution
    
    Returns:
    - Cov: filled covariance matrix
    """
    N = len(w_fft)
    w_fft_conj = w_fft.conj()
    
    # Compute upper triangle (including diagonal)
    for i in prange(N):
        for j in range(i, N):
            result = 0
            for p in range(N):
                index_i = (i - p) % N
                index_j = (j - p) % N
                index_PSD = (p + N//2) % N                
                result += w_fft[index_i] * w_fft_conj[index_j] * PSD[index_PSD]
                
            Cov[i, j] = result
            Cov[j, i] = np.conjugate(Cov[i, j])
    
    Cov *= Delta_f
    return Cov


def noise_covariance_w_pos(Sn, w, Deltat, diag = None):
    '''
    Compute N*N Fourier-domain covariance matrix for windowed process

    Fourier convention: standard, \tilde{F}(f) = \int dt e^{-2 i pi f t}F(t)

    Args:
      Sn      # Values of PSD Sn(i Deltaf), i=0,...,N/2 included (size N/2+1)
      w       # Values of window w(i Deltat), i=0,...,N-1 (size N)
      Deltat  # Value of time interval (Deltaf = 1/(N Deltat))
    '''

    N = len(w)
    M = N//2 + 1
    Deltaf = 1./(N*Deltat)


    w_tilde = np.fft.rfft(w)
    u_tilde = np.fft.fft(Sn[::-1])
    
    rangeM = np.arange(M)

    Sigma_tilde_w = np.zeros((M,M), dtype=complex)
    # Compute both upper and lower diagonals (in the circulant sense, with periodicity)
    # try:
    if diag is None:
        diag_range = range(M//2 + 1)
    else:
        diag_range = range(diag//2 + 1)

    for diag_index in diag_range:
        v = w_tilde * np.conj(np.roll(w_tilde, -diag_index))
        v_tilde = np.fft.fft(v)
        diag_vals =  (Deltaf/2) * np.fft.ifft(u_tilde * v_tilde)
        diag_vals_rolled = np.roll(diag_vals[0:M],diag_index)[::-1]
        Sigma_tilde_w[np.roll(rangeM, -diag_index), rangeM] = diag_vals_rolled

    np.fill_diagonal(Sigma_tilde_w, 0.5 * np.real(Sigma_tilde_w.diagonal()))

    Sigma_tilde_w = Sigma_tilde_w + Sigma_tilde_w.T.conj()

    return Sigma_tilde_w

def noise_covariance_modulation_analytical(Sn, A, B, Omega, Deltat):
    """
    Analytical noise covariance for modulation w(t) = A + B*cos(2*pi*Omega*t/T_obs).
    
    Matches the normalization convention of noise_covariance_w_pos.

    TODO: THIS IS BROKEN
    """
    import warnings
    
    M = len(Sn)
    N = 2 * (M - 1)
    
    k = int(np.round(Omega))
    if np.abs(Omega - k) > 1e-6:
        warnings.warn(f"Omega={Omega:.6f} not integer. Using k={k}")
    if k >= M:
        raise ValueError(f"Omega={Omega} too large (k >= M={M})")
    
    Sigma = np.zeros((M, M), dtype=float)
    
    # Normalization factor to match FFT convention
    norm = N / (2 * Deltat)
    
    # Diagonal
    Sigma[np.arange(M), np.arange(M)] = (A**2 + B**2/2.0) * norm * Sn
    
    if k > 0:
        # First sidebands
        i = np.arange(M - k)
        j = i + k
        term1 = A * B * norm * Sn[i]
        term2 = np.zeros(len(j))
        valid = j >= 2*k
        term2[valid] = (B**2/4.0) * norm * Sn[j[valid] - 2*k]
        
        Sigma[i, j] = term1 + term2
        Sigma[j, i] = term1 + term2
        
        # Second sidebands
        if 2*k < M:
            i2 = np.arange(M - 2*k)
            j2 = i2 + 2*k
            val = (B**2/4.0) * norm * Sn[i2]
            Sigma[i2, j2] = val
            Sigma[j2, i2] = val
    
    return Sigma

def regularise_matrix(Cov_Matrix, window, tol = 0.7, check_results = False):
    """
    Inputs: Cov_Matrix : Noise covariance matrix
            window : window function
            tol (arg): Essentially fraction of singular values to ignore in window
            check_results (bool): If True, then print out useful debugging techniques

    Outputs: Cov_Matrix_reg_inv : Regularised inverse
    """

    U,S,Vh = np.linalg.svd(Cov_Matrix)           # Compute SVD
    matrix_length = len(S)

    N_remove = len(np.argwhere(window < tol))//2    # Number of singular values to remove
    N_retain = len(S) - N_remove                 # Compute number of singular values to retain
    S_inv = S**-1                                # Compute inverse of singular matrix. 
    
    S_inv_regular = []
    for i in range(0,matrix_length):
        if i >= N_retain: 
            S_inv_regular.append(0)              # Set infinite singular values to zero. 
        else:
            S_inv_regular.append(S_inv[i])
    Cov_Matrix_reg_inv = Vh.T.conj() @ np.diag(S_inv_regular) @ U.conj().T
    np.fill_diagonal(Cov_Matrix_reg_inv, np.real(np.diag(Cov_Matrix_reg_inv))) # Force diagonal to be real. 

    if check_results == True:
        indices_remove = np.arange(N_retain+1, matrix_length, 1)
        fig, ax = plt.subplots(1,1, figsize = (9,5))
        ax.semilogy(S, '*', c = 'black', label = 'non truncated')
        ax.semilogy(S[0:N_retain], c = 'red', linestyle = '--', label = 'truncated')
        ax.semilogy(indices_remove, S[N_retain+1:], c = 'cyan', linestyle = '-.', label = 'Removed')
        
        ax.set_xlabel(r'Index of matrix', fontsize = 16)
        ax.set_ylabel(r'Singular Value of SVD', fontsize = 16)
        ax.set_title(r'Analysis of singular matrix', fontsize = 16)
        ax.legend(loc = "lower left", fontsize = 16)
        plt.grid()
        plt.savefig(PLOT_DIRECTORY + "/singular_values.pdf",bbox_inches = "tight")
        
        print("For a tolerance of {}".format(tol))
        print("We remove {} singular values".format(N_remove))
    return Cov_Matrix_reg_inv