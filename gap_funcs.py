import numpy as np
def Fisher_Matrix(params, Cov_Matrix_inv, N_channels = 2, w_t = 1.0, 
                    treated_stationary = True, calc_deriv = True, 
                    deriv_vec = None, return_deriv = False, 
                    **waveform_kwargs):
    """
    Computes the Fisher Matrix for the given parameters.

    Args:
        params (dict): Dictionary of waveform parameters.
        Cov_Matrix_inv (ndarray): Inverse covariance matrix.
        N_channels (int): Number of channels.
        w_t (float): Weighting factor for the window function.
        treated_stationary (bool): Flag to indicate if treated as stationary.
        return_deriv (bool): Flag to return derivative waveforms.
        **waveform_kwargs: Additional arguments for the waveform generator.

    Returns:
        ndarray or tuple: Fisher matrix, or (derivatives, Fisher matrix) if return_deriv is True.
    """
    # Compute array of parameter derivatives 

    if calc_deriv == True:
        deriv_vec = Compute_Derivative_Waveform(params, w_t = w_t,  **waveform_kwargs)

    param_label_string = list(params.keys())
    N_params = len(param_label_string) - 1   # All parameters apart from the flag Lframe

    FM_AE = [np.eye(N_params, dtype = float) for _ in range(N_channels)]

    # Both codes below will build an upper triangular matrix with (half) the usual values
    # on the main diagonals
    # If the process is treated as a stationary process, use usual stat covariance matrix
    if treated_stationary == False: 
        for i in range(N_params):
            for j in range(i, N_params):
                for k in range(N_channels):
                    if i == j:
                        FM_AE[k][i,j] = 2*np.real(0.5*deriv_vec[i][k].conj() @ Cov_Matrix_inv @ deriv_vec[j][k])
                    else:
                        FM_AE[k][i,j] = 2*np.real(deriv_vec[i][k].conj() @ Cov_Matrix_inv @ deriv_vec[j][k])
    # If the process is treated as a non-stationary process, use regularised covariance matrix
    elif treated_stationary == True:
        for i in range(N_params):
            for j in range(i, N_params):
                for k in range(N_channels):
                    if i == j:
                        FM_AE[k][i,j] = 2*np.real(0.5*xp.sum(deriv_vec[i][k].conj() * (np.diag(Cov_Matrix_inv)) * deriv_vec[j][k]))
                    else:
                        FM_AE[k][i,j] = 2*np.real(xp.sum(deriv_vec[i][k].conj() *  (np.diag(Cov_Matrix_inv)) * deriv_vec[j][k]))
    
    FM_AE = FM_AE[0] + FM_AE[1] # Combine to build FM over A and E. 
    FM_AE = (FM_AE + FM_AE.T)   # Build FM

    if return_deriv == False:
        return FM_AE
    elif return_deriv == True:
        return deriv_vec, FM_AE 

def MLE_estimate(params, noise_f, Cov_Matrix_inv, N_channels = 2, w_t = 1.0, 
                    treated_stationary = True, calc_deriv = True, 
                    deriv_vec = None,compute_FM = True,
                    FM = None, **waveform_kwargs):
    """
    Estimates the Maximum Likelihood Estimators (MLEs) for the parameters.

    Args:
        params (dict): Dictionary of waveform parameters.
        noise_f (ndarray): Frequency domain noise data.
        Cov_Matrix_inv (ndarray): Inverse covariance matrix.
        N_channels (int): Number of channels.
        w_t (float): Weighting factor for the window function.
        treated_stationary (bool): Flag to indicate if treated as stationary.
        compute_FM (bool): Flag to compute Fisher Matrix.
        FM (ndarray): Fisher Matrix (if precomputed).
        **waveform_kwargs: Additional arguments for the waveform generator.

    Returns:
        ndarray: Maximum Likelihood Estimators for the parameters.
    """

    # Simple error checking
    if compute_FM == True:
        if FM is not None:
            raise ValueError("Careful, you are providing a FM. Set to None or set compute_FM = False")
        deriv_vec, FM = Fisher_Matrix(params, Cov_Matrix_inv, N_channels = 2, w_t = w_t, 
                        treated_stationary = treated_stationary, return_deriv = True, **waveform_kwargs)
    elif compute_FM == False and calc_deriv == True:
        if FM is None:
            raise ValueError("Need to supply FM")
        if deriv_vec is None:
            raise ValueError("Need to supply derivatives")
        deriv_vec = Compute_Derivative_Waveform(params, w_t = w_t, **waveform_kwargs) 

    N_params = FM.shape[0]

    # Compute (\partial_{theta}h|n) that appears in CV formula
    if treated_stationary == False:
        inn_prod_A = xp.asarray([2*xp.real(noise_f[0].conj() @ 
                                    Cov_Matrix_inv @ 
                                    deriv_vec[i][0]) for i in range(N_params)])

        inn_prod_E = xp.asarray([2*xp.real(noise_f[1].conj() @ 
                        Cov_Matrix_inv @ 
                        deriv_vec[i][1]) for i in range(N_params)])
    elif treated_stationary == True:
        inn_prod_A = xp.asarray([2*xp.real(xp.sum(noise_f[0].conj() 
                                            * np.diag(Cov_Matrix_inv)  
                                            * deriv_vec[i][0])) for i in range(N_params)])

        inn_prod_E = xp.asarray([2*xp.real(xp.sum(noise_f[1].conj() 
                                            * np.diag(Cov_Matrix_inv)    
                                            * deriv_vec[i][1])) for i in range(N_params)])

    # Compute parameter covariance matrix
    FM_inv = np.linalg.inv(FM)
    inn_prod_AE = inn_prod_A + inn_prod_E 

    # Compute \Delta \theta 
    Fisher_Forecast_Delta_Theta = FM_inv @ inn_prod_AE # Calculate Cutler Valisneri Bias

    param_vals = np.array(list(params.values()))[:-1] # Ignore Lframe parameter

    # Compute maximum likelihood values 
    MLEs = np.real(param_vals[0:N_params] + Fisher_Forecast_Delta_Theta)
    return MLEs

def Mismodelling_FM_Matrix(params, Cov_Matrix_true_model,  
                           Cov_Matrix_mis_model_inv,
                           Cov_Matrix_true_model_inv,
                           w_t=1.0, 
                           **waveform_params):
    """
    Computes the Upsilon Matrix, representing the total parameter uncertainty due to model 
    mismodeling, and the inverse Fisher Matrix for the correct model.

    Args:
        params (dict): Dictionary of waveform parameters.
        Cov_Matrix_true_model (ndarray): Covariance matrix for the true model.
        Cov_Matrix_mis_model_inv (ndarray): Inverse covariance matrix for the mismodeling case.
        Cov_Matrix_true_model_inv (ndarray): Inverse covariance matrix for the true model.
        w_t (float): Weighting factor for the window function. Default is 1.0.
        **waveform_params: Additional arguments for the waveform generator.

    Returns:
        tuple: 
            - Upsilon_Matrix (ndarray): Matrix representing the total mismodeling parameter uncertainty.
            - gamma_correct_model_inv (ndarray): Inverse Fisher Matrix for the correct model.
    """
    
    # Calculate derivative of waveform and the Fisher matrix for the mismodeling case
    deriv_vec, gamma_mis_model = Fisher_Matrix(
        params, Cov_Matrix_mis_model_inv, N_channels=2, w_t=w_t, 
        treated_stationary=True, return_deriv=True, calc_deriv=True, 
        **waveform_params
    )

    # Compute the covariance matrix inverse for mismodeling
    mismodel_cov_matrix_inv = (
        Cov_Matrix_mis_model_inv @ Cov_Matrix_true_model @ Cov_Matrix_mis_model_inv
    )

    # Compute the inner Fisher Matrix for mismodeling
    gamma_inner = Fisher_Matrix(
        params, mismodel_cov_matrix_inv, N_channels=2, w_t=w_t, 
        treated_stationary=False, return_deriv=False, calc_deriv=False, 
        deriv_vec=deriv_vec, **waveform_params
    )                

    # Inverse of the mismodeling Fisher matrix
    gamma_mis_model_inv = np.linalg.inv(gamma_mis_model)

    # Total mismodeling parameter uncertainty
    total_mismodelling_param_uncertainty = (
        gamma_mis_model_inv @ gamma_inner @ gamma_mis_model_inv
    )

    # Fisher Matrix and its inverse for the correct model
    gamma_correct_model = Fisher_Matrix(
        params, Cov_Matrix_true_model_inv, N_channels=2, w_t=w_t, 
        treated_stationary=False, return_deriv=False, calc_deriv=False, 
        deriv_vec=deriv_vec, **waveform_params
    )
    gamma_correct_model_inv = np.linalg.inv(gamma_correct_model)

    # Upsilon Matrix representing the ratio of mismodeling uncertainty to correct model uncertainty
    Upsilon_Matrix = total_mismodelling_param_uncertainty / gamma_correct_model_inv 

    return Upsilon_Matrix, gamma_correct_model_inv