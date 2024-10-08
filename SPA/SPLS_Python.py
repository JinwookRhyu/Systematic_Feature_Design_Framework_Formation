import numpy as np
from scipy.optimize import least_squares
from sklearn.cross_decomposition import PLSRegression
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pdb

def SPLS(X, y, K, eta, kappa = 0.5, select = 'pls2', eps = 1e-4, max_steps = 200):
    """
    A Python port of the original Sparse PLS function by Chun and Keleş (doi.org/10.1111/j.1467-9868.2009.00723.x)
    """
    if K >= X.shape[0] or K > X.shape[1]:
        raise ValueError(f'K = {K} was input to SPLS(), but K must be <= {np.minimum(X.shape[0]-1, X.shape[1])}')
    # Scaling X and y by their means and stdevs - commented out here because SPA already scales the data
    """y = (y - np.mean(y)) / np.std(y, ddof = 1)
    X = (X - np.mean(X, axis = 0))
    X_std = np.std(X, ddof = 1, axis = 0) # Separate calculation to avoid divisions by 0
    X[X_std!=0] = X[X_std!=0] / X_std[X_std!=0]"""
    # Convenience variables
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    potential_idx = np.arange(X.shape[1])
    #betahat = np.zeros((X.shape[0], y.shape[0]))
    betahat = np.zeros((X.shape[1]))
    betahat_list = []
    new_selected_list = []
    y_for_PLS = y # This variable is changed in each iteration of the for loop below if select == 'pls2'
    X_for_PLS = X # This variable is changed in each iteration of the for loop below if select == 'simpls'

    for k in range(K):
        # PLS setup and PLS
        Z = X_for_PLS.T @ y_for_PLS # Matrix multiplication
        direction_vector = _SPLS_dv(Z, eta, kappa, eps, max_steps).squeeze()
        all_selected = np.unique(potential_idx[(direction_vector != 0) | (betahat != 0)]) # All selected variables
        new_selected_list.append( potential_idx[(direction_vector != 0) | (betahat == 0)] ) # Variables that were selected in this iteration of the for loop
        xA = X[:, all_selected]
        my_PLS = PLSRegression(k+1, scale = False, tol = eps).fit(X, y)
        # Updating the beta
        betahat = np.zeros((X.shape[1]))
        #pdb.set_trace()
        betahat[all_selected] = my_PLS.coef_.squeeze()[all_selected]
        betahat_list.append(betahat)
        if select.casefold() == 'pls2':
            temp = (X@betahat).reshape(-1, 1) # Reshaping to avoid broadcasting issues
            y_for_PLS = y - temp
        elif select.casefold() == 'simpls':
            proj = 'TODO' # TODO
            raise NotImplementedError('select = "simpls" has not been implemented yet. Please use select = "pls2"')

    return [betahat, all_selected, betahat_list, new_selected_list]

def _SPLS_dv(Z, eta, kappa, eps, max_steps):
    """
    Calculates the direction vector for SPLS. Automatically called by SPLS()
    """
    Z = Z / np.median(np.abs(Z)) # Normalizing by the median
    if Z.shape[1] == 1:
        circle = _UST(Z, eta)
    else:
        M = Z @ Z.T # Matrix multiplication
        circle = np.ones((Z.shape[0], 1))
        for idx in range(max_steps):
            # Calculating the A vector
            if kappa == 0.5:
                U, _, V = np.linalg.svd(M @ circle)
                U = np.atleast_2d(U[:, 0]).T # We want just the first vector, but we want a 2D Nx1 array
                A = U @ V # Matrix multiplication
            else: # TODO: not tested since SPA always calls SPLS with kappa = 0.5
                while _hfunction(eps, M, circle, kappa2) <= 1e5 and _hfunction(eps, M, circle, kappa2) * _hfunction(1e30, M, circle, kappa2) > 0:
                    M *= 2
                    circle *= 2
                # Optimizing A for a fixed circle
                lambda_s = least_squares(_hfunction, eps, args = (M, circle, kappa2), bounds = (eps, 1e30)).x[0]
                A = kappa2 * np.linalg.inv(M + lambda_s*np.identity(M.shape[0])) @ M @ circle
            # Updating the circle based on the new A vector
            current_circle = _UST(M@A, eta)
            discrepancy = np.max(np.abs(current_circle-circle))
            circle = current_circle
            if discrepancy <= eps:
                break

    return circle

def _UST(Z, eta):
    """
    Univariate soft thresholding estimator. Helper function called by _SPLS_dv() during the calculation of direction vectors
    """
    circle = np.zeros_like(Z, dtype = float)
    my_val = np.abs(Z) - eta * np.max(np.abs(Z))
    circle[my_val >= 0] = my_val[my_val >= 0] * np.sign(Z)[my_val >= 0]
    return circle

def _hfunction(lambda_val, M, circle, kappa2):
    """
    A helper function called by _SPLS_dv() during the calculation of direction vectors if kappa < 0.5
    """
    alpha = np.linalg.inv(M + lambda_val*np.identity(M.shape[0])) @ M @ circle
    obj = alpha.T @ alpha - 1/kappa2**2
    return obj

""" Temp to run R online
library(spls)
X = rbind(c(1, 3, 0), c(0, 1, 1), c(2, 1, -1))
print(X)
y = c(7, 1, 5)
oi <- spls(X, y, 2, 0.9, scale.y = TRUE)
print(attributes(oi))
print("This is betahat")
print(oi$betahat)
print("This is A")
print(oi$A)
print("This is betamat")
print(oi$betamat)
print("This is new2As")
print(oi$new2As)
print(oi)
"""
