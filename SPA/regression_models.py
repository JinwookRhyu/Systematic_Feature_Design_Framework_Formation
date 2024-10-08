"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics
"""
import statsmodels.api as sm
from SPLS_Python import SPLS
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import f_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error
import numpy as np
import numpy.matlib as matlib
import nonlinear_regression as nr
import warnings
warnings.filterwarnings("ignore") # TODO: Want to just ignore the PLS constant residual warnings, but this will do for now


def model_getter(model_name):
    '''Return the model according to the name'''
    switcher = {
            'OLS': OLS_fitting,
            'SPLS': SPLS_fitting,
            'EN': EN_fitting,
            'LASSO': LASSO_fitting,
            'ALVEN': ALVEN_fitting,
            'LCEN': LCEN_fitting,
            'RR': RR_fitting,
            'DALVEN': DALVEN_fitting,
            'DALVEN_full_nonlinear': DALVEN_fitting_full_nonlinear}
    return switcher[model_name]

def OLS_fitting(X, y, X_test, y_test):
    """
    Fits data using ordinary least squares: y = a*x1 + b*x2 + ...

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    """

    # Training
    OLS_model = sm.OLS(y, X).fit()
    yhat_train = OLS_model.predict().reshape((-1,1))
    mse_train = mean_squared_error(y, yhat_train)
    OLS_params = OLS_model.params.reshape(-1,1) # Fitted parameters
    
    # Testing
    yhat_test = OLS_model.predict(X_test).reshape((-1,1))
    mse_test = mean_squared_error(y_test, yhat_test)
    return(OLS_model, OLS_params, mse_train, mse_test, yhat_train, yhat_test)

def SPLS_fitting(X, y, X_test, y_test, K = None, eta = None, eps = 1e-4, maxstep = 1000):
    """
    Fits data using an Sparse PLS model coded in R

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    K: int, optional, default = None
        Number of latent variables
    eta: float, optional, default = None
        Sparsity tuning parameter ranging from 0 to 1
    """
    _, selected_variables, _, _ = SPLS(X, y, K, eta, eps = eps, max_steps = maxstep)
    SPLS_model = PLSRegression(K, scale = False, tol = eps).fit(X[:, selected_variables], y)
    SPLS_params = SPLS_model.coef_.squeeze()
    # Predictions and MSEs
    yhat_train = np.dot(X[:, selected_variables], SPLS_params)
    yhat_test = np.dot(X_test[:, selected_variables], SPLS_params)
    mse_train = mean_squared_error(y, yhat_train)
    mse_test = mean_squared_error(y_test, yhat_test)

    SPLS_params_full = np.zeros((X.shape[1]))
    SPLS_params_full[selected_variables] = SPLS_params

    return SPLS_model, SPLS_params_full, mse_train, mse_test, yhat_train, yhat_test

def EN_fitting(X, y, X_test, y_test, alpha, l1_ratio, max_iter = 10000, tol = 1e-4, use_cross_entropy = False):
    """
    Fits data using sklearn's Elastic Net model

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    """

    # Training
    EN_model = ElasticNet(random_state = 0, alpha = alpha, l1_ratio = l1_ratio, fit_intercept = False, max_iter = max_iter, tol = tol)
    EN_model.fit(X, y)
    yhat_train = EN_model.predict(X).reshape((-1,1))
    if use_cross_entropy:
        mse_train = 0
    else:
        mse_train = mean_squared_error(y, yhat_train)
    EN_params = EN_model.coef_.reshape((-1,1)) # Fitted parameters

    # Testing
    yhat_test = EN_model.predict(X_test).reshape((-1,1))
    if use_cross_entropy:
        mse_test = 0
    else:
        mse_test = mean_squared_error(y_test, yhat_test)
    return (EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test)

def RR_fitting(X, y, X_test, y_test, alpha, l1_ratio):
    """
    Fits data using sklearn's Ridge Regression model

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    """

    # Training
    RR_model = Ridge(alpha = alpha, fit_intercept = False).fit(X, y)
    yhat_train = np.dot(X, RR_params).reshape((-1,1))
    mse_train = mean_squared_error(y, yhat_train)
    RR_params = RR_model.coef_.reshape((-1,1)) # Fitted parameters

    # Testing
    yhat_test = np.dot(X_test, RR_params).reshape((-1,1))
    mse_test = mean_squared_error(y_test, yhat_test)
    return (RR_model, RR_params, mse_train, mse_test, yhat_train, yhat_test)

def LASSO_fitting(X, y, X_test, y_test, alpha, max_iter = 10000, tol = 1e-4):
    """
    Fits data using sklearn's LASSO model

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float
        Regularization parameter weight.
    """

    # Training
    LASSO_model = Lasso(random_state = 0, alpha = alpha, fit_intercept = False, max_iter = max_iter, tol = tol)
    LASSO_model.fit(X, y)
    yhat_train = LASSO_model.predict(X).reshape((-1,1))
    mse_train = mean_squared_error(y, yhat_train)
    LASSO_params = LASSO_model.coef_.reshape((-1,1)) # Fitted parameters

    # Testing
    yhat_test = LASSO_model.predict(X_test).reshape((-1,1))
    mse_test = mean_squared_error(y_test, yhat_test)
    return (LASSO_model, LASSO_params, mse_train, mse_test, yhat_train, yhat_test)

def LCEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree = 1, lag = 0, max_iter = 10000, tol = 1e-4, trans_type = 'all', LCEN_type = 'LCEN', selection = None, all_pos = None):
    """
    Fits data using Algebraic Learning Via Elastic Net
    https://doi.org/10.1016/j.compchemeng.2020.107103

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float or int
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    degree : int, optional, default = 1
        The degrees of nonlinear mapping.
    alpha_num : int, optional, default = None
        Penalty weight used.
    cv : bool, optional, default = False
        Whether the run is done to validate a model or test the best model.
    selection : str, optional, default = 'p_value'
        Selection criterion for the pre-processing step
        Must be in {'p_value', 'percentage', 'elbow'}
    select_value : float, optional, default = 0.10
        The minimum p_value for a variable to be considered relevant (when selection == 'p_value'), ...
            or the first select_value percent variables to be used (when selection == 'percentage').
        Not relevant when selection == 'elbow'
    trans_type : str, optional, default = 'all'
        Feature transformation based on LCEN ('all') or polynomial ('poly')
    use_cross_entropy : bool, optional, default = False
        Whether to use cross entropy or MSE for model comparison.
    """

    # Feature transformation
    if trans_type == 'all':
        X, X_test, label_names = _feature_trans(X, X_test, degree, interaction = True, trans_type = trans_type, all_pos = all_pos)
    elif trans_type == 'poly':
        X, X_test = nr.poly_feature(X, X_test, degree = degree, interaction = True, power = True)
    else:
        raise ValueError(f'trans_type must be "all" or "poly", but you passed {trans_type}')

    # Scale data (based on z-score)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(X)
    X = scaler_x.transform(X)
    X_test = scaler_x.transform(X_test)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)



    if selection is not None:
        X = X[:, selection]
        X_test = X_test[:, selection]

    if X.shape[1] == 0:
        LCEN_model = None
        LCEN_params = np.empty(0)
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)  # Ok to be 0 because y is scaled --> mean(y) is approximately 0
        yhat_test = np.zeros(y_test.shape)
    else:
        LCEN_model, LCEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(X, y, X_test, y_test, alpha,
                                                                                           l1_ratio, max_iter, tol)
        # Calculating information criteria values
    num_train = X.shape[0]  # Typically written as n
    num_parameter = (LCEN_params != 0).flatten().sum()  # Typically written as k
    AIC = num_train * np.log(
        mse_train) + 2 * num_parameter  # num_train * log(MSE) is one of the formulae to replace L. Shown in https://doi.org/10.1002/wics.1460, for example.
    AICc = AIC + 2 * num_parameter * (num_parameter + 1) / (num_train - num_parameter - 1)
    BIC = num_train * np.log(mse_train) + num_parameter * np.log(num_train)

    yhat_train = scaler_y.inverse_transform(yhat_train.reshape(-1, 1))
    yhat_test = scaler_y.inverse_transform(yhat_test.reshape(-1, 1))

    return (LCEN_model, LCEN_params, mse_train, mse_test, yhat_train, yhat_test, label_names, (AIC, AICc, BIC))

def DALVEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree, lag, alpha_num = None, cv = False, max_iter = 10000, 
                tol = 1e-4, selection = 'p_value', select_value = 0.10, trans_type = 'all', use_cross_entropy = False):
    """
    Fits data using Dynamic Algebraic Learning Via Elastic Net
    https://doi.org/10.1016/j.compchemeng.2020.107103

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float or int
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    lag : int
        Variable lags to be considered. A lag of L will make the model take into account ...
            variables from point xt to xt-L and from yt to yt-L
    degree : int
        The degrees of nonlinear mapping.
    alpha_num : int, optional, default = None
        Penalty weight used.
    cv : bool, optional, default = False
        Whether the run is done to validate a model or test the best model.
    selection : str, optional, default = 'p_value'
        Selection criterion for the pre-processing step
        Must be in {'p_value', 'percentage', 'elbow'}
    select_value : float, optional, default = 0.10
        The minimum p_value for a variable to be considered relevant (when selection == 'p_value'), ...
            or the first select_value percent variables to be used (when selection == 'percentage').
        Not relevant when selection == 'elbow'
    trans_type : str, optional, default = 'all'
        Feature transformation based on ALVEN ('all') or polynomial ('poly')
    use_cross_entropy : bool, optional, default = False
        Whether to use cross entropy or MSE for model comparison.
    """

    # Feature transformation
    if trans_type == 'all':
        X, X_test = _feature_trans(X, X_test, degree = degree, interaction = 'later')
    elif trans_type == 'poly':
        X, X_test = nr.poly_feature(X, X_test, degree = degree, interaction = True, power = True)
    else:
        raise ValueError(f'trans_type must be "all" or "poly", but you passed {trans_type}')

    # Lag padding for X
    XD = X[lag:]
    XD_test = X_test[lag:]
    for i in range(lag):
        XD = np.hstack((XD, X[lag-1-i : -i-1]))
        XD_test = np.hstack((XD_test, X_test[lag-1-i : -i-1]))
    # Lag padding for y in design matrix
    for i in range(lag):
        XD = np.hstack((XD, y[lag-1-i : -i-1]))
        XD_test = np.hstack((XD_test, y_test[lag-1-i : -i-1]))    
    # Shorterning y
    y = y[lag:]
    y_test = y_test[lag:]

    # Remove features with insignificant variance
    sel = VarianceThreshold(threshold = tol).fit(XD)
    XD = sel.transform(XD)
    XD_test = sel.transform(XD_test)

    # Scale data (based on z-score)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(XD)
    XD = scaler_x.transform(XD)
    XD_test = scaler_x.transform(XD_test)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)

    # Eliminate features
    f_test, p_values = f_regression(XD, y.flatten())

    if selection == 'p_value':
        retain_index = p_values < select_value
    elif selection == 'percentage':
        number = int(np.ceil(select_value * XD.shape[1]))
        f_test.sort()
        value = f_test[-number]
        retain_index = f_test >= value
    else:
        f = np.copy(f_test)
        f = np.sort(f)[::-1]

        axis = np.linspace(0,len(f)-1, len(f))
        AllCord = np.concatenate((axis.reshape(-1,1), f.reshape(-1,1)), axis = 1)

        lineVec = AllCord[-1] - AllCord[0]
        lineVec = lineVec/ np.sqrt(np.sum(lineVec**2))
        vecFromFirst = AllCord- AllCord[0] # Distance from each point to the line
        # And calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis = 1) # np.repeat(np.atleast_2d(lineVec), len(f), 0)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine**2, axis = 1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        retain_index = f_test >= value

    XD_fit =  XD[:, retain_index]
    XD_test_fit = XD_test[:, retain_index]

    if XD_fit.shape[1] == 0:
        print('No variable was selected by ALVEN')
        DALVEN_model = None
        DALVEN_params = None
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)
        yhat_test = np.zeros(y_test.shape)
        alpha = 0
    else:
        if alpha_num is not None and cv:
            XD_max = np.concatenate((XD_fit, XD_test_fit), axis = 0)
            y_max = np.concatenate((y, y_test), axis = 0)
            alpha_max = (np.sqrt(np.sum(np.dot(XD_max.T, y_max)**2, axis = 1)).max()) / XD_max.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(XD_fit.T, y) ** 2, axis=1)).max()) / XD_fit.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        #EN for model fitting
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(XD_fit, y, XD_test_fit, y_test, alpha, l1_ratio, max_iter, tol, use_cross_entropy)

        num_train = XD_fit.shape[0]
        num_parameter = sum(DALVEN_params!=0)[0]
        AIC = num_train*np.log(mse_train) + 2*num_parameter
        AICc = num_train*np.log(mse_train) + (num_parameter+num_train)/(1-(num_parameter+2)/num_train) # TODO: Fix the divide by zero errors
        BIC = num_train*np.log(mse_train) + num_parameter*np.log(num_train)
    return (DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index, (AIC,AICc,BIC))

def DALVEN_fitting_full_nonlinear(X, y, X_test, y_test, alpha, l1_ratio, degree, lag, alpha_num = None, cv = False, max_iter = 10000, 
                                tol = 1e-4, selection = 'p_value', select_value = 0.10, trans_type = 'all', use_cross_entropy = False):
    """
    Fits data using Dynamic Algebraic Learning Via Elastic Net - full non-linear mapping
    https://doi.org/10.1016/j.compchemeng.2020.107103

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float or int
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    lag : int
        Variable lags to be considered. A lag of L will make the model take into account ...
            variables from point xt to xt-L and from yt to yt-L
    degree : int
        The degrees of nonlinear mapping.
    alpha_num : int, optional, default = None
        Penalty weight used.
    cv : bool, optional, default = False
        Whether the run is done to validate a model or test the best model.
    selection : str, optional, default = 'p_value'
        Selection criterion for the pre-processing step
        Must be in {'p_value', 'percentage', 'elbow'}
    select_value : float, optional, default = 0.10
        The minimum p_value for a variable to be considered relevant (when selection == 'p_value'), ...
            or the first select_value percent variables to be used (when selection == 'percentage').
        Not relevant when selection == 'elbow'
    trans_type : str, optional, default = 'all'
        Feature transformation based on ALVEN ('all') or polynomial ('poly')
    use_cross_entropy : bool, optional, default = False
        Whether to use cross entropy or MSE for model comparison.
    """
    # Lag padding for X
    XD = X[lag:]
    XD_test = X_test[lag:]
    for i in range(lag):
        XD = np.hstack((XD, X[lag-1-i : -i-1]))
        XD_test = np.hstack((XD_test, X_test[lag-1-i : -i-1]))
    # Lag padding for y in design matrix
    for i in range(lag):
        XD = np.hstack((XD, y[lag-1-i : -i-1]))
        XD_test = np.hstack((XD_test, y_test[lag-1-i : -i-1]))
    # Shorterning y
    y = y[lag:]
    y_test = y_test[lag:]

    # Feature transformation
    if trans_type == 'all':
        XD, XD_test = nr.feature_trans(XD, XD_test, degree = degree, interaction = 'later')
    else:
        XD, XD_test = nr.poly_feature(XD, XD_test, degree = degree, interaction = True, power = True)
  
    # Remove features with insignificant variance
    sel = VarianceThreshold(threshold = tol).fit(XD)
    XD = sel.transform(XD)
    XD_test = sel.transform(XD_test)

    # Scale data (based on z-score)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(XD)
    XD = scaler_x.transform(XD)
    XD_test = scaler_x.transform(XD_test)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)

    # Eliminate features
    f_test, p_values = f_regression(XD, y.flatten())

    if selection == 'p_value':
        retain_index = p_values < select_value
    elif selection == 'percentage':
        number = int(np.ceil(select_value * XD.shape[1]))
        f_test.sort()
        value = f_test[-number]
        retain_index = f_test >= value
    else:
        f = np.copy(f_test)
        f = np.sort(f)[::-1]

        axis = np.linspace(0,len(f)-1, len(f))
        AllCord = np.concatenate((axis.reshape(-1,1), f.reshape(-1,1)), axis = 1)

        lineVec = AllCord[-1] - AllCord[0]
        lineVec = lineVec/ np.sqrt(np.sum(lineVec**2))
        vecFromFirst = AllCord- AllCord[0] # Distance from each point to the line
        # And calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis = 1) # np.repeat(np.atleast_2d(lineVec), len(f), 0)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine**2, axis = 1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        retain_index = f_test >= value
 
    XD_fit =  XD[:, retain_index]
    XD_test_fit = XD_test[:, retain_index]

    if XD_fit.shape[1] == 0:
        print('No variable was selected by ALVEN')
        DALVEN_model = None
        DALVEN_params = None
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)
        yhat_test = np.zeros(y_test.shape)
        alpha = 0
    else:
        if alpha_num is not None and cv:
            XD_max = np.concatenate((XD_fit, XD_test_fit), axis = 0)
            y_max = np.concatenate((y, y_test), axis = 0)
            alpha_max = (np.sqrt(np.sum(np.dot(XD_max.T,y_max)**2, axis = 1)).max()) / XD_max.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(XD_fit.T, y)**2, axis = 1)).max()) / XD_fit.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        #EN for model fitting
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(XD_fit, y, XD_test_fit, y_test, alpha, l1_ratio, max_iter, tol, use_cross_entropy)

        num_train = XD_fit.shape[0]
        num_parameter = sum(DALVEN_params!=0)[0]
        AIC = num_train*np.log(mse_train) + 2*num_parameter
        AICc = num_train*np.log(mse_train) + (num_parameter+num_train)/(1-(num_parameter+2)/num_train)
        BIC = num_train*np.log(mse_train) + num_parameter*np.log(num_train)
    return (DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index, (AIC,AICc,BIC))


def ALVEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree=1, alpha_num=None, cv=False, max_iter=10000,
                  tol=1e-4, selection='p_value', select_value=0.15, trans_type='all', use_cross_entropy=False, all_pos = None):
    """
    Fits data using Algebraic Learning Via Elastic Net
    https://doi.org/10.1016/j.compchemeng.2020.107103

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float or int
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    degree : int, optional, default = 1
        The degrees of nonlinear mapping.
    alpha_num : int, optional, default = None
        Penalty weight used.
    cv : bool, optional, default = False
        Whether the run is done to validate a model or test the best model.
    selection : str, optional, default = 'p_value'
        Selection criterion for the pre-processing step
        Must be in {'p_value', 'percentage', 'elbow'}
    select_value : float, optional, default = 0.10
        The minimum p_value for a variable to be considered relevant (when selection == 'p_value'), ...
            or the first select_value percent variables to be used (when selection == 'percentage').
        Not relevant when selection == 'elbow'
    trans_type : str, optional, default = 'all'
        Feature transformation based on ALVEN ('all') or polynomial ('poly')
    use_cross_entropy : bool, optional, default = False
        Whether to use cross entropy or MSE for model comparison.
    """

    # Feature transformation
    if trans_type == 'all':
        X, X_test, label_names = _feature_trans(X, X_test, degree, interaction=True, trans_type=trans_type,
                                                all_pos=all_pos)
    elif trans_type == 'poly':
        X, X_test = nr.poly_feature(X, X_test, degree=degree, interaction=True, power=True)
    else:
        raise ValueError(f'trans_type must be "all" or "poly", but you passed {trans_type}')

    # Remove features with insignificant variance
    sel = VarianceThreshold(threshold=tol).fit(X)
    X = sel.transform(X)
    X_test = sel.transform(X_test)
    label_names = sel.transform(label_names.reshape(1,-1))[0]

    # Scale data (based on z-score)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(X)
    X = scaler_x.transform(X)
    X_test = scaler_x.transform(X_test)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)

    # Eliminate features
    f_test, p_values = f_regression(X, y.flatten())

    if selection == 'p_value':
        retain_index = p_values < select_value
    elif selection == 'percentage':
        number = int(np.ceil(select_value * X.shape[1]))
        f_test.sort()
        value = f_test[-number]
        retain_index = f_test >= value
    else:
        f = np.copy(f_test)
        f = np.sort(f)[::-1]

        axis = np.linspace(0, len(f) - 1, len(f))
        AllCord = np.concatenate((axis.reshape(-1, 1), f.reshape(-1, 1)), axis=1)

        lineVec = AllCord[-1] - AllCord[0]
        lineVec /= np.sqrt(np.sum(lineVec ** 2))
        vecFromFirst = AllCord - AllCord[0]  # Distance from each point to the line
        # And calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        retain_index = f_test >= value

    X_fit = X[:, retain_index]
    X_test_fit = X_test[:, retain_index]
    label_names = label_names[retain_index]

    if X_fit.shape[1] == 0:
        print('No variable was selected by ALVEN')
        ALVEN_model = None
        ALVEN_params = None
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)
        yhat_test = np.zeros(y_test.shape)
        alpha = 0
    else:
        if alpha_num is not None and cv:
            X_max = np.concatenate((X_fit, X_test_fit), axis=0)
            y_max = np.concatenate((y, y_test), axis=0)
            alpha_max = (np.sqrt(np.sum(np.dot(X_max.T, y_max) ** 2, axis=1)).max()) / X_max.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(X_fit.T, y) ** 2, axis=1)).max()) / X_fit.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        # EN for model fitting
        ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(X_fit, y, X_test_fit, y_test,
                                                                                           alpha, l1_ratio, max_iter,
                                                                                           tol, use_cross_entropy)

        yhat_train = scaler_y.inverse_transform(yhat_train)
        yhat_test = scaler_y.inverse_transform(yhat_test)

    return (ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index, label_names)

def _feature_trans(X, X_test = None, degree = 2, interaction = True, trans_type = 'all', all_pos = None):
    """
    A helper function that is automatically called by SPA.
    Performs non-linear transformations of X (and X_test). Transformations include polynomial transforms up to "degree", ...
    interactions between the raw variables in X (up to "degree" powers at the same time). If trans_type == 'all', also ...
    includes ln, sqrt, and inverse transforms of power up to "degree". Also includes some interaction terms among them.

    Parameters
    ----------
    X, y : Numpy array with shape N x m
        Training data predictors.
    X_test, y_test : Numpy array with shape N_test x m, optional, default = None
        Testing data predictors.
    degree : integer, optional, default = 2
        The highest degree used for polynomial and interaction transforms.
        For example, degree = 1 would include x0 to xN. Degree = 2 would also include (x0)^2 to (xN)^2...
            and, if interaction == True, (x0*x1), (x0*x2), ..., (x0*xN), ..., (x1*xN) terms.
    interaction : boolean, optional, default = True
        Whether to include polynomial interaction terms up to "degree". For example, degree = 2 interactions include ...
            terms of the x0*x1 form. Degree = 3 interactions also include terms of the x2*x3*x5 and (x1)^2 * x4 form
    trans_type : str in {'all', 'poly', 'simple_interaction'}, optional, default == 'all'
        Whether to include all transforms (polynomial, log, sqrt, and inverse), only polynomial transforms (and, ...
            optionally, interactions), or just interactions.
        The log, sqrt, and inverse terms never include interactions among the same transform type (such as ln(x1)*ln(x4)), but ...
            include some interactions among each other for the same variable (such as ln(x0)*1/x0, x0*sqrt(x0), etc.).
    """
    X_test_out = None # Declaring this variable to avoid errors when doing the return statement
    # Setting up the transforms
    Xlog = np.where(X!=0, np.log(np.abs(X)), -50) # Avoiding log(0) = -inf
    if all_pos is None:
        if X_test is not None: # Columns where all entries are >= 0 for sqrt
            all_pos = np.all(X >= 0, axis = 0) & np.all(X_test >= 0, axis = 0)
        else:
            all_pos = np.all(X >= 0, axis = 0)
    Xsqrt = np.sqrt(X[:, all_pos])
    Xinv = 1/X
    Xinv[Xinv >= 1e15] = 1e15
    Xinv[Xinv <= -1e15] = -1e15
    if X_test is not None:
        Xlog_t = np.where(X_test!=0, np.log(np.abs(X_test)), -50) # Avoiding log(0) = -inf
        Xsqrt_t = np.sqrt(X_test[:, all_pos])
        Xinv_t = 1/X_test
        Xinv_t[Xinv_t >= 1e15] = 1e15
        Xinv_t[Xinv_t <= -1e15] = -1e15

    # Polynomial transforms (and interaction terms)
    poly = PolynomialFeatures(degree, include_bias = True, interaction_only = trans_type.casefold() == 'simple_interaction')
    X_out = poly.fit_transform(X)
    label_names = poly.get_feature_names_out()
    interaction_column = np.array([' ' in elem for elem in label_names], dtype = bool) # To filter out interactions if user asked for a polynomial-only transform. Also for the log, sqrt, and inv terms below when poly_trans_only == False
    for idx in range(len(label_names)):
        label_names[idx] = label_names[idx].translate({ord(i): '*' for i in ' '}) # str.translate replaces the characters on the right of the for (in this case, a whitespace) with an asterisk
    if X_test is not None:
        X_test_out = poly.fit_transform(X_test)
    # Discarding the interaction terms (x1*x2, x2*x3*x5, (x1)^2 * x4, etc.) if requested to do so by the user
    if not interaction:
        X_out = X_out[:, ~interaction_column]
        label_names = label_names[~interaction_column]
        if X_test is not None:
            X_test_out = X_test_out[:, ~interaction_column]
    # Including ln, sqrt, and inverse terms; and also their higher-degree transforms if degree >= 2
    if trans_type.casefold() == 'all':
        # ln transform
        Xlog_trans = poly.fit_transform(Xlog)[:, ~interaction_column][:, 1:] # Transforming and removing interactions, as we do not care about log-log interactions
        X_out = np.column_stack((X_out, Xlog_trans))
        temp_label_names = poly.get_feature_names_out()[~interaction_column][1:] # [1:] to remove the bias term, as it was already included with the polynomial transformations
        for idx in range(len(temp_label_names)): # Converting the names from x0-like to ln(x0)-like
            power_split = temp_label_names[idx].split('^') # Separates the variable (e.g.: x0 or x1) from the power it was raised to, if it exists
            if len(power_split) > 1:
                power = f'^{power_split[-1]}'
                base = ''.join(power_split[:-1])
            else: # Variable hasn't beed raised to any power (equivalent to ^1), but we need an empty "power" variable to avoid errors
                power = ''
                base = power_split[0]
            temp_label_names[idx] = f'ln({base}){power}' # Final name is of the form ln(x1)^3, not ln(x1^3)
        label_names = np.concatenate((label_names, temp_label_names))
        if X_test is not None:
            Xlog_test_trans = poly.fit_transform(Xlog_t)[:, ~interaction_column][:, 1:]
            X_test_out = np.column_stack((X_test_out, Xlog_test_trans))
        # sqrt transform
        X_out = np.column_stack((X_out, Xsqrt))
        temp_label_names = [f'sqrt(x{number})' for number in range(X.shape[1]) if all_pos[number]]
        label_names = np.concatenate((label_names, temp_label_names))
        if X_test is not None:
            X_test_out = np.column_stack((X_test_out, Xsqrt_t))
        # Inverse transform
        Xinv_trans = poly.fit_transform(Xinv)[:, 1:] # [:, 1:] to remove the bias term, as it was already included with the polynomial transformations
        X_out = np.column_stack((X_out, Xinv_trans))
        temp_label_names = poly.get_feature_names_out()[1:] # [1:] to remove the bias term, as it was already included with the polynomial transformations
        for idx in range(len(temp_label_names)): # Converting the names from x0-like to ln(x0)-like
            temp_label_names[idx] = temp_label_names[idx].translate({ord(i): '*' for i in ' '}) # str.translate replaces the characters on the right of the for (in this case, a whitespace) with an asterisk
            temp_label_names[idx] = f'1/({temp_label_names[idx]})' # 1/(x1^3) is the same as (1/x1)^3, so we do not need the fancy manipulations used above in the ln transform naming
        label_names = np.concatenate((label_names, temp_label_names))
        if X_test is not None:
            Xinv_test_trans = poly.fit_transform(Xinv_t)[:, 1:]
            X_test_out = np.column_stack((X_test_out, Xinv_test_trans))
        # Specific interactions between X, ln(X), sqrt(X), and 1/X that occur for degree >= 2
        if degree >= 2:
            normal_plus_half_interaction = np.column_stack([X[:, all_pos]**(pow1+0.5) for pow1 in range(1, degree)])
            normal_plus_half_names = [f'x{number}^{pow1+0.5}' for pow1 in range(1, degree) for number in range(X.shape[1]) if all_pos[number]]
            log_inv_interaction = np.column_stack([Xlog**pow1 * Xinv**pow2 for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree])
            log_inv_names = [f'ln(x{number})^{pow1}/(x{number})^{pow2}' for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree for number in range(X.shape[1])]
            log_inv_names = [elem[:-2].replace('^1/', '/') + elem[-2:].replace('^1', '') for elem in log_inv_names] # Removing ^1. String addition to avoid removing ^10, ^11, ^12, ...
            inv_minus_half_interaction = np.column_stack([Xinv[:, all_pos]**(pow1-0.5) for pow1 in range(1, degree)])
            inv_minus_half_names = [f'1/(x{number}^{pow1-0.5})' for pow1 in range(1, degree) for number in range(X.shape[1]) if all_pos[number]]
            if degree == 2: # degree == 2 does not have the ln(x) * sqrt(x) / x type of interactions
                X_out = np.column_stack((X_out, normal_plus_half_interaction, log_inv_interaction, inv_minus_half_interaction))
                label_names = np.concatenate((label_names, normal_plus_half_names, log_inv_names, inv_minus_half_names))
            else:
                log_inv_minus_oneandhalf_interaction = np.column_stack([Xlog[:, all_pos]**pow1 * Xinv[:, all_pos]**(pow2-0.5) for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree-1])
                log_inv_minus_oneandhalf_names = [f'ln(x{number})^{pow1}/(x{number}^{pow2-0.5})' for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree-1 for number in range(X.shape[1]) if all_pos[number]]
                log_inv_minus_oneandhalf_names = [elem[:-2].replace('^1/', '/') + elem[-2:].replace('^1', '') for elem in log_inv_minus_oneandhalf_names] # Removing ^1. String addition to avoid removing ^10, ^11, ^12, ...
                X_out = np.column_stack((X_out, normal_plus_half_interaction, log_inv_interaction, inv_minus_half_interaction, log_inv_minus_oneandhalf_interaction))
                label_names = np.concatenate((label_names, normal_plus_half_names, log_inv_names, inv_minus_half_names, log_inv_minus_oneandhalf_names))
            if X_test is not None:
                normal_plus_half_interaction = np.column_stack([X_test[:, all_pos]**(pow1+0.5) for pow1 in range(1, degree)])
                log_inv_interaction = np.column_stack([Xlog_t**pow1 * Xinv_t**pow2 for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree])
                inv_minus_half_interaction = np.column_stack([Xinv_t[:, all_pos]**(pow1-0.5) for pow1 in range(1, degree)])
                if degree == 2:
                    X_test_out = np.column_stack((X_test_out, normal_plus_half_interaction, log_inv_interaction, inv_minus_half_interaction))
                else:
                    log_inv_minus_oneandhalf_interaction = np.column_stack([Xlog_t[:, all_pos]**pow1 * Xinv_t[:, all_pos]**(pow2-0.5) for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree-1])
                    X_test_out = np.column_stack((X_test_out, normal_plus_half_interaction, log_inv_interaction, inv_minus_half_interaction, log_inv_minus_oneandhalf_interaction))

    X_out = X_out[:, 1:]
    if X_test is not None:
        X_test_out = X_test_out[:, 1:]
    label_names = label_names[1:]

    return (X_out, X_test_out, label_names)