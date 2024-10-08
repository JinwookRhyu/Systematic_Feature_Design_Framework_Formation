"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics
"""
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit, TimeSeriesSplit, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils._testing import ignore_warnings
import regression_models as rm
import nonlinear_regression as nr
import nonlinear_regression_other as nro
from itertools import product
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

def CVpartition(X, y, Type = 'Re_KFold', K = 5, Nr = 10, random_state = 0, group = None):
    """
    Partitions data for cross-validation and bootstrapping.
    Returns a generator with the split data.

    Parameters
    ----------
    model_name : str
        Which model to use
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    Type : str, optional, default = 'Re_KFold'
        Which cross validation method to use.
    K : int, optional, default = 5
        Number of folds used in cross validation.
    Nr : int, optional, default = 10
        Number of CV repetitions used when cv_type in {'MC', 'Re_KFold', 'GroupShuffleSplit'}.
    random_state : int, optional, default = 0
        Seed used for the random number generator.
    group : list, optional, default = None
        Group indices for grouped CV methods.
    """
    Type = Type.casefold() # To avoid issues with uppercase/lowercase
    if Type == 'mc':
        CV = ShuffleSplit(n_splits = Nr, test_size = 1/K, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'single':
        X, X_test, y, y_test = train_test_split(X, y, test_size = 1/K, random_state = random_state)
        yield (X, y, X_test, y_test)
    elif Type == 'kfold':
        CV = KFold(n_splits = int(K))
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 're_kfold':
        CV = RepeatedKFold(n_splits = int(K), n_repeats = Nr, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'timeseries':
        TS = TimeSeriesSplit(n_splits = int(K))
        for train_index, val_index in TS.split(X):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'single_group':
        label = np.unique(group)
        num = int(len(label)/K)
        final_list = np.squeeze(group == label[0])
        for i in range(1, num):
            final_list = final_list | np.squeeze(group == label[i])
        yield(X[~final_list], y[~final_list], X[final_list], y[final_list])
    elif Type == 'group':
        label = np.unique(group)
        for i in range(len(label)):
            yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])], y[np.squeeze(group == label[i])])
    elif Type == 'group_no_extrapolation':
        label = np.unique(group)
        for i in range(len(label)):
            if min(label) < label[i] and label[i] < max(label):
                yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])], y[np.squeeze(group == label[i])])
    elif Type == 'groupkfold':
        gkf = GroupKFold(n_splits = int(K))
        for train_index, val_index in gkf.split(X, y, groups = group):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'groupshufflesplit':
        gss = GroupShuffleSplit(n_splits = int(Nr), test_size = 1 / K, random_state = random_state)
        for train_index, val_index in gss.split(X, y, groups = group):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'no_cv':
        yield (X, y, X, y)
    elif Type == 'single_ordered':
        num = X.shape[0]
        yield (X[:num-round(X.shape[0]*1/K):], y[:num-round(X.shape[0]*1/K):], X[num-round(X.shape[0]*1/K):], y[num-round(X.shape[0]*1/K):])
    else:
        raise ValueError(f'{Type} is not a valid CV type.')

def CV_mse(model_name, X, y, X_test, y_test, cv_type = 'Re_KFold', K_fold = 5, Nr = 10, eps = 1e-4, alpha_num = 20, group = None, **kwargs):
    """
    Determines the best hyperparameters using MSE based on information criteria.
    Also returns MSE and yhat data for the chosen model.

    Parameters
    ----------
    model_name : str
        Which model to use
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    cv_type : str, optional, default = None
        Which cross validation method to use.
    K_fold : int, optional, default = 5
        Number of folds used in cross validation.
    Nr : int, optional, default = 10
        Number of CV repetitions used when cv_type in {'MC', 'Re_KFold', 'GroupShuffleSplit'}.
    eps : float, optional, default = 1e-4
        Tolerance. TODO: expand on this.
    alpha_num : int, optional, default = 20
        Penalty weight used when model_name in {'RR', 'EN', 'ALVEN', 'DALVEN', 'DALVEN_full_nonlinear'}.
    **kwargs : dict, optional
        Non-default hyperparameters for model fitting.
    """

    # For alpha_max
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(X)
    X_scale = scaler_x.transform(X)
    X_test_scale = scaler_x.transform(X_test)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y_scale = scaler_y.transform(y)
    y_test_scale = scaler_y.transform(y_test)

    if 'robust_priority' not in kwargs:  # This should not be the case unless the user called this function manually, which is not recommended
        kwargs['robust_priority'] = False
    if 'l1_ratio' not in kwargs:
        kwargs['l1_ratio'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
    if 'alpha' not in kwargs:  # Unusual scenario, since SPA passes a default kwargs['alpha'] == 20
        kwargs['alpha'] = np.concatenate(([0], np.logspace(-4.3, 0, 20)))
    elif isinstance(kwargs['alpha'], int):  # User passed an integer instead of a list of values
        kwargs['alpha'] = np.concatenate(([0], np.logspace(-4.3, 0, kwargs['alpha'])))
    if 'use_cross_entropy' not in kwargs:
        kwargs['use_cross_entropy'] = False


    if model_name == 'EN':
        EN = rm.model_getter(model_name)
        if 'use_cross_entropy' not in kwargs:
            kwargs['use_cross_entropy'] = False

        RMSE_result = np.empty((len(kwargs['l1_ratio']) * alpha_num, K_fold * Nr)) * np.nan
        Var = np.empty((len(kwargs['l1_ratio']) * alpha_num, K_fold * Nr)) * np.nan

        hyperparam_prod = list(product(kwargs['l1_ratio'], range(alpha_num)))
        print(f'There are {len(hyperparam_prod)} hyperparameter combinations')

        with Parallel(n_jobs=-1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(
                    CVpartition(X, y, Type=cv_type, K=K_fold, Nr=Nr, group=group)):
                temp = PAR(delayed(_EN_joblib_fun)(X_train, y_train, X_val, y_val, eps, kwargs,
                                                     prod_idx, this_prod) for prod_idx, this_prod in
                           enumerate(hyperparam_prod))
                RMSE_result[:, counter], Var[:, counter] = zip(*temp)
        print('')

        RMSE_mean = np.nanmean(RMSE_result, axis=1)
        ind = np.nanargmin(RMSE_mean)
        if kwargs['robust_priority']:
            RMSE_std = np.nanstd(RMSE_result, axis=1)
            RMSE_min = RMSE_mean[ind]
            RMSE_bar = RMSE_min + RMSE_std[ind]
            Var_num = np.nansum(Var, axis=1)
            ind = np.nonzero(Var_num == np.nanmin(Var_num[RMSE_mean < RMSE_bar]))  # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = ind[0][0]

        # Hyperparameter setup
        l1_ratio = hyperparam_prod[ind][0]
        alpha = kwargs['alpha'][hyperparam_prod[ind][1]]
        hyperparams = {}
        hyperparams['alpha'] = alpha
        hyperparams['l1_ratio'] = l1_ratio

        # Fit the final model
        if l1_ratio == 0:
            EN_model = Ridge(alpha = alpha, fit_intercept = False).fit(X_scale, y_scale)
            EN_params= EN_model.coef_.reshape(-1,1)
            yhat_train = scaler_y.inverse_transform(EN_model.predict(X_scale).reshape(-1,1))
            yhat_test = scaler_y.inverse_transform(EN_model.predict(X_test_scale).reshape(-1,1))
            rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
            rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])
        else:
            EN_model, EN_params, rmse_train, rmse_test, yhat_train, yhat_test = EN(X_scale, y_scale, X_test_scale, y_test_scale, alpha = alpha, l1_ratio = l1_ratio)
            yhat_train = scaler_y.inverse_transform(yhat_train.reshape(-1,1))
            yhat_test = scaler_y.inverse_transform(yhat_test.reshape(-1,1))
            rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
            rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])

        return(hyperparams, EN_model, EN_params, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'PLS':
        if not(cv_type.startswith('Group')) and 'K' not in kwargs: # For non-grouped CV types
            kwargs['K'] = np.linspace(1, min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)) )
        elif 'K' not in kwargs:
            kwargs['K'] = np.linspace(1, min(X.shape[1], X.shape[0]-1), min(X.shape[1], X.shape[0]-1))

        RMSE_result = np.empty((len(kwargs['K']), K_fold * Nr)) * np.nan

        for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group=group)):
            scaler_x_train = StandardScaler(with_mean=True, with_std=True)
            scaler_x_train.fit(X_train)
            X_train_scale = scaler_x_train.transform(X_train)
            X_val_scale = scaler_x_train.transform(X_val)
            scaler_y_train = StandardScaler(with_mean=True, with_std=True)
            scaler_y_train.fit(y_train)
            y_train_scale = scaler_y_train.transform(y_train)
            y_val_scale = scaler_y_train.transform(y_val)

            for i in range(min(len(kwargs['K']), X_train.shape[0]-1)):
                PLS = PLSRegression(scale = False, n_components = int(kwargs['K'][i]), tol = eps).fit(X_train_scale, y_train_scale)
                PLS_para = PLS.coef_.reshape(-1,1)
                yhat_val = np.dot(X_val_scale, PLS_para)
                RMSE_result[i, counter] = np.sqrt(np.sum((yhat_val.reshape(-1,1)-y_val_scale)**2)/y_val_scale.shape[0])

        RMSE_mean = np.nanmean(RMSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(RMSE_mean), RMSE_mean.shape)
        if kwargs['robust_priority']:
            RMSE_std = np.nanstd(RMSE_result, axis = 1)
            RMSE_min = RMSE_mean[ind]
            RMSE_bar = RMSE_min + RMSE_std[ind]
            ind = np.nonzero(kwargs['K'] == np.nanmin(kwargs['K'][RMSE_mean < RMSE_bar])) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = ind[0][0]

        # Hyperparameter setup
        hyperparams = {}
        hyperparams['K'] = kwargs['K'][ind]

        # Fit the final model
        PLS_model = PLSRegression(scale = False, n_components = int(hyperparams['K'])).fit(X_scale,y_scale)
        PLS_params = PLS_model.coef_.reshape(-1,1)
        yhat_train = scaler_y.inverse_transform(np.dot(X_scale, PLS_params).reshape(-1,1))
        yhat_test = scaler_y.inverse_transform(np.dot(X_test_scale, PLS_params).reshape(-1,1))
        rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
        rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])

        return(hyperparams, PLS_model, PLS_params, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'SPLS':
        SPLS = rm.SPLS_fitting
        if not(cv_type.startswith('Group')) and 'K' not in kwargs: # For non-grouped CV types
            kwargs['K'] = np.linspace(1, min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), dtype = np.uint64)
        elif 'K' not in kwargs:
            kwargs['K'] = np.linspace(1, min(X.shape[1], X.shape[0]-1), min(X.shape[1], X.shape[0]-1), dtype = np.uint64)

        if 'eta' not in kwargs:
            kwargs['eta'] = np.linspace(0, 1, 20, endpoint = False)[::-1] #eta = 0 -> use normal PLS

        RMSE_result = np.empty((len(kwargs['K']) * len(kwargs['eta']), K_fold * Nr)) * np.nan
        Var = np.empty((len(kwargs['K']) * len(kwargs['eta']), K_fold * Nr)) * np.nan

        hyperparam_prod = list(product(kwargs['K'], kwargs['eta']))
        print(f'There are {len(hyperparam_prod)} hyperparameter combinations')

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                temp = PAR(delayed(_SPLS_joblib_fun)(X_train, y_train, X_val, y_val, eps, alpha_num, kwargs, counter,
                        prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                RMSE_result[:, counter], Var[:, counter] = zip(*temp)
        print('')

        RMSE_mean = np.nanmean(RMSE_result, axis=1)
        ind = np.nanargmin(RMSE_mean)
        if kwargs['robust_priority']:
            RMSE_std = np.nanstd(RMSE_result, axis=1)
            RMSE_min = RMSE_mean[ind]
            RMSE_bar = RMSE_min + RMSE_std[ind]
            Var_num = np.nansum(Var, axis=1)
            ind = np.nonzero(Var_num == np.nanmin(Var_num[RMSE_mean < RMSE_bar]))  # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = ind[0][0]

        # Hyperparameter setup
        K = hyperparam_prod[ind][0]
        eta = hyperparam_prod[ind][1]
        hyperparams = {}
        hyperparams['K'] = K
        hyperparams['eta'] = eta

        # Fit the final model
        SPLS_model, SPLS_params, rmse_train, rmse_test, yhat_train, yhat_test = SPLS(X_scale, y_scale, X_test_scale, y_test_scale, eta = eta, K = K)
        yhat_train = scaler_y.inverse_transform(yhat_train.reshape(-1,1))
        yhat_test = scaler_y.inverse_transform(yhat_test.reshape(-1,1))
        rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
        rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])

        return(hyperparams, SPLS_model, SPLS_params, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'LASSO': # TODO: add onestd?
        LASSO = rm.model_getter(model_name)
        if 'alpha' not in kwargs:
            alpha_max = (np.sqrt(np.sum(np.dot(X_scale.T,y_scale) ** 2, axis=1)).max())/X.shape[0]
            kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), alpha_num)[::-1]

        RMSE_result = np.empty((len(kwargs['alpha']), K_fold * Nr)) * np.nan

        for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
            scaler_x_train = StandardScaler(with_mean=True, with_std=True)
            scaler_x_train.fit(X_train)
            X_train_scale = scaler_x_train.transform(X_train)
            X_val_scale = scaler_x_train.transform(X_val)
            scaler_y_train = StandardScaler(with_mean=True, with_std=True)
            scaler_y_train.fit(y_train)
            y_train_scale = scaler_y_train.transform(y_train)
            y_val_scale = scaler_y_train.transform(y_val)

            for i in range(len(kwargs['alpha'])):
                _, _, _, _, _, yhat_val = LASSO(X_train_scale, y_train_scale, X_val_scale, y_val_scale, alpha = kwargs['alpha'][i])
                RMSE_result[i, counter] = np.sqrt(np.sum((yhat_val.reshape(-1,1)-y_val_scale)**2)/y_val_scale.shape[0])

        RMSE_mean = np.nanmean(RMSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(RMSE_mean), RMSE_mean.shape)

        # Hyperparameter setup
        alpha = kwargs['alpha'][ind]
        hyperparams = {}
        hyperparams['alpha'] = alpha

        # Fit the final model
        LASSO_model, LASSO_params, rmse_train, rmse_test, yhat_train, yhat_test = LASSO(X_scale, y_scale, X_test_scale, y_test_scale, alpha = alpha)
        return(hyperparams, LASSO_model, LASSO_params, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'POLY': # TODO: add onestd?
        OLS = rm.model_getter('OLS')
        if 'degree' not in kwargs:
            kwargs['degree'] = [2,3,4]
        if 'interaction' not in kwargs:
            kwargs['interaction'] = True
        if 'power' not in kwargs:
            kwargs['power'] = True

        RMSE_result = np.empty((len(kwargs['degree']), K_fold * Nr)) * np.nan

        for d in range(len(kwargs['degree'])):
            X_trans, _ = nr.poly_feature(X, X_test, degree = kwargs['degree'][d], interaction = kwargs['interaction'], power = kwargs['power'])
            X_trans = np.hstack((np.ones([X_trans.shape[0], 1]), X_trans))
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_trans, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                _, _, _, mse, _, _ = OLS(X_train, y_train, X_val, y_val)
                RMSE_result[d, counter] = mse

        RMSE_mean = np.nanmean(RMSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(RMSE_mean), RMSE_mean.shape)
        """if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            Var_num = np.nansum(Var, axis = 1)
            ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) ) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = (ind[0][0], ind[1][0])"""

        # Hyperparameter setup
        degree = kwargs['degree'][ind[0]]
        hyperparams = {}
        hyperparams['degree'] = int(degree)

        # Fit the final model
        X_trans, X_trans_test = nr.poly_feature(X, X_test, degree = degree, interaction = kwargs['interaction'], power = kwargs['power'])
        X_trans = np.hstack((np.ones([X_trans.shape[0], 1]), X_trans))
        X_trans_test = np.hstack((np.ones([X_trans_test.shape[0], 1]), X_trans_test))

        POLY_model, POLY_params, rmse_train, rmse_test, yhat_train, yhat_test = OLS(X_trans, y, X_trans_test, y_test)
        return(hyperparams, POLY_model, POLY_params, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'PLS':
        if not(cv_type.startswith('Group')) and 'K' not in kwargs: # For non-grouped CV types
            kwargs['K'] = np.linspace(1, min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)) )
        elif 'K' not in kwargs:
            kwargs['K'] = np.linspace(1, min(X.shape[1], X.shape[0]-1), min(X.shape[1], X.shape[0]-1))

        RMSE_result = np.empty((len(kwargs['K']), K_fold * Nr)) * np.nan

        for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group=group)):
            scaler_x_train = StandardScaler(with_mean=True, with_std=True)
            scaler_x_train.fit(X_train)
            X_train_scale = scaler_x_train.transform(X_train)
            X_val_scale = scaler_x_train.transform(X_val)
            scaler_y_train = StandardScaler(with_mean=True, with_std=True)
            scaler_y_train.fit(y_train)
            y_train_scale = scaler_y_train.transform(y_train)
            y_val_scale = scaler_y_train.transform(y_val)

            for i in range(min(len(kwargs['K']), X_train.shape[0]-1)):
                PLS = PLSRegression(scale = False, n_components = int(kwargs['K'][i]), tol = eps).fit(X_train_scale, y_train_scale)
                PLS_para = PLS.coef_.reshape(-1,1)
                yhat_val = np.dot(X_val_scale, PLS_para)
                RMSE_result[i, counter] = np.sqrt(np.sum((yhat_val.reshape(-1,1)-y_val_scale)**2)/y_val_scale.shape[0])

        RMSE_mean = np.nanmean(RMSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(RMSE_mean), RMSE_mean.shape)
        if kwargs['robust_priority']:
            RMSE_std = np.nanstd(RMSE_result, axis = 1)
            RMSE_min = RMSE_mean[ind]
            RMSE_bar = RMSE_min + RMSE_std[ind]
            ind = np.nonzero(kwargs['K'] == np.nanmin(kwargs['K'][RMSE_mean < RMSE_bar])) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = ind[0][0]

        # Hyperparameter setup
        hyperparams = {}
        hyperparams['K'] = kwargs['K'][ind]

        # Fit the final model
        PLS_model = PLSRegression(scale = False, n_components = int(hyperparams['K'])).fit(X_scale,y_scale)
        PLS_params = PLS_model.coef_.reshape(-1,1)
        yhat_train = scaler_y.inverse_transform(np.dot(X_scale, PLS_params).reshape(-1,1))
        yhat_test = scaler_y.inverse_transform(np.dot(X_test_scale, PLS_params).reshape(-1,1))
        rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
        rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])

        return(hyperparams, PLS_model, PLS_params, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'RR':
        if 'alpha' not in kwargs:
            alpha_max = (np.sqrt(np.sum(np.dot(X_scale.T,y_scale) ** 2, axis=1)).max())/X.shape[0]/0.0001
            kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps/100), np.log10(alpha_max), alpha_num)[::-1]

        RMSE_result = np.empty((len(kwargs['alpha']), K_fold * Nr)) * np.nan

        for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
            scaler_x_train = StandardScaler(with_mean=True, with_std=True)
            scaler_x_train.fit(X_train)
            X_train_scale = scaler_x_train.transform(X_train)
            X_val_scale = scaler_x_train.transform(X_val)
            scaler_y_train = StandardScaler(with_mean=True, with_std=True)
            scaler_y_train.fit(y_train)
            y_train_scale = scaler_y_train.transform(y_train)
            y_val_scale = scaler_y_train.transform(y_val)

            for i in range(len(kwargs['alpha'])):
                RR = Ridge(alpha = kwargs['alpha'][i], fit_intercept = False).fit(X_train_scale, y_train_scale)
                Para = RR.coef_.reshape(-1,1)
                yhat_val = np.dot(X_val_scale, Para)
                RMSE_result[i, counter] = np.sqrt(np.sum((yhat_val.reshape(-1,1)-y_val_scale)**2)/y_val_scale.shape[0])

        RMSE_mean = np.nanmean(RMSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(RMSE_mean), RMSE_mean.shape)
        if kwargs['robust_priority']:
            RMSE_std = np.nanstd(RMSE_result, axis = 1)
            RMSE_min = RMSE_mean[ind]
            RMSE_bar = RMSE_min + RMSE_std[ind]
            ind = np.nonzero(kwargs['alpha'] == np.nanmax(kwargs['alpha'][RMSE_mean < RMSE_bar])) # Hyperparams with the highest alpha but still within one stdev of the best MSE
            ind = ind[0][0]

        # Hyperparameter setup
        hyperparams = {}
        hyperparams['alpha'] = kwargs['alpha'][ind]

        # Fit the final model
        RR_model = Ridge(alpha = hyperparams['alpha'], fit_intercept = False).fit(X_scale,y_scale)
        RR_params = RR_model.coef_.reshape(-1,1)
        yhat_train = scaler_y.inverse_transform(np.dot(X_scale, RR_params).reshape(-1,1))
        yhat_test = scaler_y.inverse_transform(np.dot(X_test_scale, RR_params).reshape(-1,1))
        rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
        rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])
        return(hyperparams, RR_model, RR_params, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'LCEN':
        kwargs['model_name'] = model_name
        if 'degree' not in kwargs:
            kwargs['degree'] = [1, 2, 3]
        if 'label_name' not in kwargs:  # Whether to auto-generate label names for the variables [x1, x2, ..., log(x1), ..., 1/x1, ..., x1*x2, etc.]
            kwargs['label_name'] = True  # TODO: currently unused. Not sure whether I'll re-implement custom naming
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'all'

        kwargs['lag'] = [0]

        # First run for variable selection using a L1_ratio of 1 (that is, only using an L1 penalty)
        kwargs['selection'] = None
        hyperparam_prod = list(product(kwargs['degree'], [1], kwargs['alpha'], kwargs['lag']))
        print(f'Beginning variable selection runs. There are {len(hyperparam_prod)} hyperparameter combinations')
        all_pos = np.all(X >= 0, axis = 0) & np.all(X_test >= 0, axis = 0)
        with Parallel(n_jobs=-1) as PAR:
            if 'IC' in cv_type:  # Information criterion
                temp = PAR(delayed(_LCEN_joblib_fun)(X, y, X_test, y_test, eps, kwargs, all_pos, prod_idx, this_prod) for
                           prod_idx, this_prod in enumerate(hyperparam_prod))
                temp = list(zip(*temp))[
                    2]  # To isolate the (AIC, AICc, BIC) tuple, which is the 3rd subentry of each entry in the original temp
                temp = np.array(temp)
                if cv_type == 'AICc':
                    IC_result = temp[:, 1]
                elif cv_type == 'BIC':
                    IC_result = temp[:, 2]
                else:  # AIC
                    IC_result = temp[:, 0]
                ind = np.argmin(IC_result)
            else:  # Cross-validation
                RMSE_result = np.empty(
                    (len(kwargs['degree']) * len(kwargs['alpha']) * len(kwargs['lag']), K_fold * Nr)) * np.nan
                for counter, (X_train, y_train, X_val, y_val) in enumerate(
                        CVpartition(X, y, Type=cv_type, K=K_fold, Nr=Nr, group=group)):
                    temp = PAR(delayed(_LCEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, kwargs, all_pos,
                                                          prod_idx, this_prod) for prod_idx, this_prod in
                               enumerate(hyperparam_prod))
                    RMSE_result[:, counter], _, _ = zip(*temp)
                # Best hyperparameters for the preliminary run
                RMSE_mean = np.nanmean(RMSE_result, axis=1)
                ind = np.nanargmin(RMSE_mean)
        # Run to obtain the coefficients when LCEN is run with L1_ratio = 1
        degree, l1_ratio, alpha, lag = hyperparam_prod[ind]
        _, LCEN_params, _, _, _, _, label_names, _ = rm.LCEN_fitting(X, y, X_test, y_test, alpha, 1, degree, lag,
                                                                       tol=eps,
                                                                       trans_type=kwargs['trans_type'],
                                                                       LCEN_type=kwargs['model_name'], selection=None)
        kwargs['selection'] = np.abs(
            LCEN_params.flatten()) >= 2.4e-3  # TODO: This value could (should?) depend on the "eps" parameter, but I need to learn more about how it works in other models

        # Second run with a free L1_ratio but fixed degree and lag
        hyperparam_prod = list(
            product([degree], kwargs['l1_ratio'], kwargs['alpha'], [lag]))  # Degree and lag have been fixed above
        print(f'Beginning real runs. There are {len(hyperparam_prod)} hyperparameter combinations')
        with Parallel(n_jobs=-1) as PAR:
            if 'IC' in cv_type:  # Information criterion
                temp = PAR(delayed(_LCEN_joblib_fun)(X, y, X_test, y_test, eps, kwargs, all_pos, prod_idx, this_prod) for
                           prod_idx, this_prod in enumerate(hyperparam_prod))
                temp = list(zip(*temp))[
                    2]  # To isolate the (AIC, AICc, BIC) tuple, which is the 3rd subentry of each entry in the original temp
                temp = np.array(temp)
                if cv_type == 'AICc':
                    IC_result = temp[:, 1]
                elif cv_type == 'BIC':
                    IC_result = temp[:, 2]
                else:  # AIC
                    IC_result = temp[:, 0]
                ind = np.argmin(IC_result)
            else:
                RMSE_result = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold * Nr)) * np.nan
                Var = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']),
                                K_fold * Nr)) * np.nan  # Used when robust_priority == True
                for counter, (X_train, y_train, X_val, y_val) in enumerate(
                        CVpartition(X, y, Type=cv_type, K=K_fold, Nr=Nr, group=group)):
                    temp = PAR(delayed(_LCEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, kwargs, all_pos, prod_idx,
                                                          this_prod, counter) for prod_idx, this_prod in
                               enumerate(hyperparam_prod))
                    RMSE_result[:, counter], Var[:, counter], _ = zip(*temp)
                # Best hyperparameters
                RMSE_mean = np.nanmean(RMSE_result, axis=1)
                ind = np.nanargmin(RMSE_mean)
                if kwargs['robust_priority']:
                    MSE_std = np.nanstd(RMSE_result, axis=1)
                    MSE_min = RMSE_mean[ind]
                    MSE_bar = MSE_min + MSE_std[ind]
                    Var_num = np.nansum(Var, axis=1)
                    ind = np.nonzero(Var_num == np.nanmin(Var_num[
                                                              RMSE_mean < MSE_bar]))  # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
                    ind = ind[0][0]

        # Hyperparameter setup
        degree, l1_ratio, alpha, lag = hyperparam_prod[ind]
        hyperparams = {'degree': degree, 'l1_ratio': l1_ratio, 'alpha': alpha, 'lag': lag}
        # Final run with the test set and best hyperparameters
        LCEN_model, LCEN_params, rmse_train, rmse_test, yhat_train, yhat_test, label_names, ICs = rm.LCEN_fitting(X, y,
                                                                                                                    X_test,
                                                                                                                    y_test,
                                                                                                                    alpha,
                                                                                                                    l1_ratio,
                                                                                                                    degree,
                                                                                                                    lag,
                                                                                                                    tol=eps,
                                                                                                                    trans_type=
                                                                                                                   kwargs[
                                                                                                                       'trans_type'],
                                                                                                                    LCEN_type=
                                                                                                                   kwargs[
                                                                                                                       'model_name'],
                                                                                                                    selection=
                                                                                                                   kwargs[
                                                                                                                       'selection'])
        rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
        rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])
        # Unscaling the model coefficients as per stackoverflow.com/questions/23642111/how-to-unscale-the-coefficients-from-an-lmer-model-fitted-with-a-scaled-respon
        X, _, _ = rm._feature_trans(X, degree=degree, trans_type=kwargs['trans_type'], all_pos = all_pos)
        LCEN_params_unscaled = LCEN_params.squeeze()
        label_names = label_names[kwargs['selection']]
        # Removing the features that had coefficients equal to zero after the final model selection
        final_selection = np.abs(LCEN_params.squeeze()) >= 1e-3  # TODO: again, the eps dependency
        LCEN_params_unscaled = LCEN_params_unscaled[final_selection].reshape(
            -1)  # Reshape to avoid 0D arrays when only one variable is selected
        label_names = label_names[final_selection].reshape(-1)
        LCEN_params_unscaled = list(zip(label_names, LCEN_params_unscaled))
        if 'IC' in cv_type:
            return (
                hyperparams, LCEN_model, LCEN_params_unscaled, rmse_train, rmse_test, yhat_train, yhat_test, IC_result[ind])
        else:
            return (
                hyperparams, LCEN_model, LCEN_params_unscaled, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])
    elif model_name == 'ALVEN':
        # ALVEN = rm.model_getter(model_name)
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1, 2, 3]
        if 'label_name' not in kwargs:
            kwargs['label_name'] = True
        if 'selection' not in kwargs:
            kwargs['selection'] = 'p_value'
        if 'select_value' not in kwargs:
            kwargs['select_value'] = 0.10
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        if 'use_cross_entropy' not in kwargs:
            kwargs['use_cross_entropy'] = False

        RMSE_result = np.empty((len(kwargs['degree']) * alpha_num * len(kwargs['l1_ratio']), K_fold * Nr)) * np.nan
        Var = np.empty((len(kwargs['degree']) * alpha_num * len(kwargs['l1_ratio']),
                        K_fold * Nr)) * np.nan  # Used when robust_priority == True
        hyperparam_prod = list(product(kwargs['degree'], kwargs['l1_ratio'], range(alpha_num)))
        print(f'There are {len(hyperparam_prod)} hyperparameter combinations')

        with Parallel(n_jobs=-1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(
                    CVpartition(X, y, Type=cv_type, K=K_fold, Nr=Nr, group=group)):
                temp = PAR(delayed(_ALVEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, alpha_num, kwargs, counter,
                                                      prod_idx, this_prod) for prod_idx, this_prod in
                           enumerate(hyperparam_prod))
                RMSE_result[:, counter], Var[:, counter] = zip(*temp)
        print('')

        RMSE_mean = np.nanmean(RMSE_result, axis=1)
        ind = np.nanargmin(RMSE_mean)
        if kwargs['robust_priority']:
            RMSE_std = np.nanstd(RMSE_result, axis=1)
            RMSE_min = RMSE_mean[ind]
            RMSE_bar = RMSE_min + RMSE_std[ind]
            Var_num = np.nansum(Var, axis=1)
            ind = np.nonzero(Var_num == np.nanmin(Var_num[
                                                      RMSE_mean < RMSE_bar]))  # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = ind[0][0]

        # Hyperparameter setup
        degree = hyperparam_prod[ind][0]
        l1_ratio = hyperparam_prod[ind][1]
        alpha = hyperparam_prod[ind][2]

        ALVEN_model, ALVEN_params, rmse_train, rmse_test, yhat_train, yhat_test, alpha, retain_index, label_names = rm.ALVEN_fitting(
            X, y, X_test, y_test, alpha,
            l1_ratio, degree, alpha_num, tol=eps, cv=False, selection=kwargs['selection'],
            select_value=kwargs['select_value'], trans_type=kwargs['trans_type'])
        hyperparams = {}
        hyperparams['alpha'] = alpha
        hyperparams['l1_ratio'] = l1_ratio
        hyperparams['degree'] = degree
        hyperparams['retain_index'] = retain_index

        rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
        rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])
        return (hyperparams, ALVEN_model, ALVEN_params, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind],
                label_names)

    elif model_name == 'RF':
        if 'max_depth' not in kwargs:
            kwargs['max_depth'] = [2, 3, 5, 10, 15, 20, 40]
        if 'n_estimators' not in kwargs:
            kwargs['n_estimators'] = [100]
        if 'min_samples_leaf' not in kwargs:
            kwargs['min_samples_leaf'] = [1]#0.02,0.05, 0.1] #, 0.05 ,0.1, 0.2] # 0.3, 0.4]

        RMSE_result = np.empty((len(kwargs['max_depth']), len(kwargs['n_estimators']), len(kwargs['min_samples_leaf']), K_fold * Nr))
        if kwargs['robust_priority']:
            # TODO: is this scoring system correct? It ignores the actual values of the paramters, caring only about their lengths and positions in the array.
            S = np.zeros((len(kwargs['max_depth']), len(kwargs['n_estimators']), len(kwargs['min_samples_leaf'])))
            for i in range(len(kwargs['max_depth'])):
                for j in range(len(kwargs['n_estimators'])):
                    for k in range(len(kwargs['min_samples_leaf'])):
                        S[i, j, k] = i/len(kwargs['max_depth']) - k/len(kwargs['min_samples_leaf'])

        for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
            scaler_x_train = StandardScaler(with_mean=True, with_std=True)
            scaler_x_train.fit(X_train)
            X_train_scale = scaler_x_train.transform(X_train)
            X_val_scale = scaler_x_train.transform(X_val)
            scaler_y_train = StandardScaler(with_mean=True, with_std=True)
            scaler_y_train.fit(y_train)
            y_train_scale = scaler_y_train.transform(y_train)
            y_val_scale = scaler_y_train.transform(y_val)

            for i in range(len(kwargs['max_depth'])):
                for j in range(len(kwargs['n_estimators'])):
                    for k in range(len(kwargs['min_samples_leaf'])):
                        _, _, _, _, yhat_val = nro.RF_fitting(X_train_scale, y_train_scale, X_val_scale, y_val_scale, kwargs['n_estimators'][j], kwargs['max_depth'][i], kwargs['min_samples_leaf'][k])
                        RMSE_result[i, j, k, counter] = np.sqrt(np.sum((yhat_val.reshape(-1,1)-y_val_scale)**2)/y_val_scale.shape[0])

        RMSE_mean = np.nanmean(RMSE_result, axis = 3)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(RMSE_mean), RMSE_mean.shape)
        if kwargs['robust_priority']:
            RMSE_std = np.nanstd(RMSE_result, axis = 3)
            RMSE_min = RMSE_mean[ind]
            RMSE_bar = RMSE_min + RMSE_std[ind]
            ind = np.nonzero(S == np.nanmin(S[RMSE_mean < RMSE_bar])) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = (ind[0][0], ind[1][0], ind[2][0])

        # Hyperparameter setup
        max_depth = kwargs['max_depth'][ind[0]]
        n_estimators = kwargs['n_estimators'][ind[1]]
        min_samples_leaf = kwargs['min_samples_leaf'][ind[2]]
        hyperparams = {}
        hyperparams['max_depth'] = max_depth
        hyperparams['n_estimators'] = n_estimators
        hyperparams['min_samples_leaf'] = min_samples_leaf

        # Fit the final model
        RF_model, rmse_train, rmse_test, yhat_train, yhat_test = nro.RF_fitting(X_scale, y_scale, X_test_scale, y_test_scale, n_estimators, max_depth, min_samples_leaf)
        yhat_train = scaler_y.inverse_transform(yhat_train.reshape(-1,1))
        yhat_test = scaler_y.inverse_transform(yhat_test.reshape(-1,1))
        rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
        rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])
        return(hyperparams, RF_model, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'XGB':
        if 'max_depth' not in kwargs:
            kwargs['max_depth'] = [2, 3, 5, 10, 15, 20, 40]
        if 'reg_lambda' not in kwargs:
            kwargs['reg_lambda'] = [1, 2.5, 3, 4, 6]
        if 'gamma' not in kwargs:
            kwargs['gamma'] = 0
        if 'n_estimators' not in kwargs:
            kwargs['n_estimators'] = 100

        RMSE_result = np.empty((len(kwargs['max_depth']) * len(kwargs['reg_lambda']), K_fold * Nr)) * np.nan
        hyperparam_prod = list(product(kwargs['max_depth'], kwargs['reg_lambda']))
        print(f'There are {len(hyperparam_prod)} hyperparameter combinations')

        with Parallel(n_jobs=-1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                temp = PAR(delayed(_XGB_joblib_fun)(X_train, y_train, X_val, y_val, kwargs, prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                RMSE_result[:, counter] = temp

        RMSE_mean = np.nanmean(RMSE_result, axis=1)
        ind = np.nanargmin(RMSE_mean)

        # Hyperparameter setup
        max_depth, reg_lambda = hyperparam_prod[ind]
        hyperparams = {'max_depth': max_depth, 'reg_lambda': reg_lambda, 'n_estimators': 100, 'gamma': 0}

        # Fit the final model
        XGB_model, rmse_train, rmse_test, yhat_train, yhat_test = nro.XGB_fitting(X_scale, y_scale, X_test_scale, y_test_scale, max_depth, reg_lambda, 0, 100)
        yhat_train = scaler_y.inverse_transform(yhat_train.reshape(-1,1))
        yhat_test = scaler_y.inverse_transform(yhat_test.reshape(-1,1))
        rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
        rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])
        return(hyperparams, XGB_model, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'SVR':
        if 'C' not in kwargs:
            kwargs['C'] = [0.001, 0.01, 0.1, 1, 10 ,50, 100, 500]
        if 'gamma' not in kwargs:
            gd = 1/X.shape[1]
            kwargs['gamma'] = [gd/50, gd/10, gd/5, gd/2, gd, gd*2, gd*5, gd*10, gd*50]
        if 'epsilon' not in kwargs:
            kwargs['epsilon'] = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3]

        RMSE_result = np.empty((len(kwargs['C']), len(kwargs['gamma']), len(kwargs['epsilon']), K_fold * Nr)) * np.nan
        if kwargs['robust_priority']:
            # TODO: is this scoring system correct? It ignores the actual values of the paramters, caring only about their lengths and positions in the array.
            S = np.zeros((len(kwargs['C']), len(kwargs['gamma']), len(kwargs['epsilon'])))
            for i in range(len(kwargs['C'])):
                for j in range(len(kwargs['gamma'])):
                    for k in range(len(kwargs['epsilon'])):
                        S[i, j, k] = i/len(kwargs['C']) - j/len(kwargs['gamma']) - k/len(kwargs['epsilon'])

        for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
            scaler_x_train = StandardScaler(with_mean=True, with_std=True)
            scaler_x_train.fit(X_train)
            X_train_scale = scaler_x_train.transform(X_train)
            X_val_scale = scaler_x_train.transform(X_val)
            scaler_y_train = StandardScaler(with_mean=True, with_std=True)
            scaler_y_train.fit(y_train)
            y_train_scale = scaler_y_train.transform(y_train)
            y_val_scale = scaler_y_train.transform(y_val)

            for i in range(len(kwargs['C'])):
                for j in range(len(kwargs['gamma'])):
                    for k in range(len(kwargs['epsilon'])):
                        _, _, _, _, yhat_val = nro.SVR_fitting(X_train_scale, y_train_scale, X_val_scale, y_val_scale, kwargs['C'][i], kwargs['epsilon'][k], kwargs['gamma'][j])
                        RMSE_result[i, j, k, counter] = np.sqrt(np.sum((yhat_val.reshape(-1,1)-y_val_scale)**2)/y_val_scale.shape[0])

        RMSE_mean = np.nanmean(RMSE_result, axis = 3)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(RMSE_mean), RMSE_mean.shape)
        if kwargs['robust_priority']:
            RMSE_std = np.nanstd(RMSE_result, axis = 3)
            RMSE_min = RMSE_mean[ind]
            RMSE_bar = RMSE_min + RMSE_std[ind]
            ind = np.nonzero(S == np.nanmin(S[RMSE_mean < RMSE_bar])) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = (ind[0][0], ind[1][0], ind[2][0])

        # Hyperparameter setup
        C = kwargs['C'][ind[0]]
        gamma = kwargs['gamma'][ind[1]]
        epsilon = kwargs['epsilon'][ind[2]]
        hyperparams = {}
        hyperparams['C'] = C
        hyperparams['gamma'] = gamma
        hyperparams['epsilon'] = epsilon

        # Fit the final model
        SVR_model, rmse_train, rmse_test, yhat_train, yhat_test =  nro.SVR_fitting(X_scale, y_scale, X_test_scale, y_test_scale, C, epsilon, gamma)
        yhat_train = scaler_y.inverse_transform(yhat_train.reshape(-1,1))
        yhat_test = scaler_y.inverse_transform(yhat_test.reshape(-1,1))
        rmse_train = np.sqrt(np.sum((yhat_train - y) ** 2) / y.shape[0])
        rmse_test = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])
        return(hyperparams, SVR_model, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind])

    elif model_name == 'DALVEN' or model_name == 'DALVEN_full_nonlinear':
        DALVEN = rm.model_getter(model_name)
        kwargs['model_name'] = model_name
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1, 2, 3]
        if 'lag' not in kwargs:
            kwargs['lag'] =  [i+1 for i in range(40)]
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        if 'selection' not in kwargs:
            kwargs['selection'] = 'p_value'
        if 'select_value' not in kwargs:
            kwargs['select_value'] = 0.05
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        if 'use_cross_entropy' not in kwargs:
            kwargs['use_cross_entropy'] = False

        hyperparam_prod = list(product(kwargs['degree'], kwargs['l1_ratio'], range(alpha_num), kwargs['lag']))
        print(f'There are {len(hyperparam_prod)} hyperparameter combinations')

        with Parallel(n_jobs = -1) as PAR:
            if 'IC' in cv_type: # Information criterion
                temp = PAR(delayed(_DALVEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, alpha_num, kwargs, counter,
                        prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                if cv_type == 'AICc':
                    IC_result = zip(*temp)[2][1]
                elif cv_type == 'BIC':
                    IC_result = zip(*temp)[2][2]
                else: # AIC
                    IC_result = zip(*temp)[2][0]
                # Min IC value (first occurrence)
                ind = np.argmin(IC_result)
            else: # Cross-validation
                RMSE_result = np.empty((len(kwargs['degree']) * alpha_num * len(kwargs['l1_ratio']) * len(kwargs['lag']), K_fold * Nr)) * np.nan
                Var = np.empty((len(kwargs['degree']) * alpha_num * len(kwargs['l1_ratio']) * len(kwargs['lag']), K_fold*Nr)) * np.nan
                for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                    temp = PAR(delayed(_DALVEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, alpha_num, kwargs, counter,
                            prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                    RMSE_result[:, counter], Var[:, counter], _ = zip(*temp)

                RMSE_mean = np.nanmean(RMSE_result, axis = 1)
                ind = np.nanargmin(RMSE_mean)
                if kwargs['robust_priority']:
                    RMSE_std = np.nanstd(RMSE_result, axis = 1)
                    RMSE_min = RMSE_mean[ind]
                    RMSE_bar = RMSE_min + RMSE_std[ind]
                    Var_num = np.nansum(Var, axis = 1)
                    ind = np.nonzero(Var_num == np.nanmin(Var_num[RMSE_mean < RMSE_bar])) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
                    ind = ind[0][0]
        print('')

        # Hyperparameter setup
        degree = hyperparam_prod[ind][0]
        l1_ratio = hyperparam_prod[ind][1]
        alpha = hyperparam_prod[ind][2]
        lag = hyperparam_prod[ind][3]

        DALVEN_model, DALVEN_params, rmse_train, rmse_test, yhat_train, yhat_test, alpha, retain_index, _ = DALVEN(X, y, X_test, y_test, alpha,
                                                                                                                   l1_ratio, degree, lag, alpha_num, tol = eps, cv = False, selection = kwargs['selection'],
                                                                                                                   select_value = kwargs['select_value'], trans_type = kwargs['trans_type'])
        hyperparams = {}
        hyperparams['alpha'] = alpha
        hyperparams['l1_ratio'] = l1_ratio
        hyperparams['degree'] = degree
        hyperparams['lag'] = lag
        hyperparams['retain_index'] = retain_index

        # Names for the retained variables(?)
        if kwargs['label_name'] :
            if model_name == 'DALVEN': # DALVEN does transform first, then lag
                if kwargs['trans_type'] == 'auto':
                    Xt, _ = nr.feature_trans(X, degree = degree, interaction = 'later')
                else:
                    Xt, _ = nr.poly_feature(X, degree = degree, interaction = True, power = True)

                # Lag padding for X
                XD = Xt[lag:]
                for i in range(lag):
                    XD = np.hstack((XD, Xt[lag-1-i : -i-1]))
                # Lag padding for y in design matrix
                for i in range(lag):
                    XD = np.hstack((XD, y[lag-1-i : -i-1]))
            else: # DALVEN_full_nonlinear does lag first, then transform
                # Lag padding for X
                XD = X[lag:]
                for i in range(lag):
                    XD = np.hstack((XD, X[lag-1-i : -i-1]))
                # Lag padding for y in design matrix
                for i in range(lag):
                    XD = np.hstack((XD, y[lag-1-i : -i-1]))

                if kwargs['trans_type'] == 'auto':
                    XD, _ = nr.feature_trans(XD, degree = degree, interaction = 'later')
                else:
                    XD, _ = nr.poly_feature(XD, degree = degree, interaction = True, power = True)

            # Remove features with insignificant variance
            sel = VarianceThreshold(threshold=eps).fit(XD)

            if 'xticks' in kwargs:
                list_name = kwargs['xticks']
            else:
                list_name = [f'x{i}' for i in range(1, np.shape(X)[1]+1)]

            list_name_final = list_name[:] # [:] makes a copy
            if kwargs['trans_type'] == 'auto':
                list_name_final += [f'log({name})' for name in list_name] + [f'sqrt({name})' for name in list_name] + [f'1/{name}' for name in list_name]

                if degree >= 2:
                    for i in range(X.shape[1]-1):
                        for j in range(i+1, X.shape[1]):
                            list_name_final += [f'{list_name[i]}*{list_name[j]}']
                    list_name_final += [f'{name}^2' for name in list_name] + [f'(log{name})^2' for name in list_name] + [f'1/{name}^2' for name in list_name] + (
                            [f'{name}^1.5' for name in list_name] + [f'log({name})/{name}' for name in list_name]+ [f'1/{name}^0.5' for name in list_name] )

                if degree >= 3:
                    for i in range(X.shape[1]-2):
                        for j in range(i+1, X.shape[1]-1):
                            for k in range(j+1, X.shape[1]):
                                list_name_final += [f'{list_name[i]}*{list_name[j]}*{list_name[k]}']
                    list_name_final += [f'{name}^3' for name in list_name] + [f'(log{name})^3' for name in list_name] + [f'1/{name}^3' for name in list_name] + (
                                [f'{name}^2.5' for name in list_name] + [f'(log{name})^2/{name}' for name in list_name]+ [f'log({name})/sqrt({name})' for name in list_name] +
                                [f'log({name})/{name}^2' for name in list_name] + [f'{name}^-1.5' for name in list_name] )

            elif degree >= 2:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        list_name_final += [f'{list_name[i]}*{list_name[j]}']
                if degree >= 3:
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            for k in range(j, X.shape[1]):
                                list_name_final += [f'{list_name[i]}*{list_name[j]}*{list_name[k]}']

            list_copy = list_name_final[:]
            for i in range(lag):
                list_name_final += [f'{s}(t-{i+1})' for s in list_copy]
            for i in range(lag):
                list_name_final += [f'y(t-{i+1})'] 

            index = list(sel.get_support())
            list_name_final = [x for x, y in zip(list_name_final, index) if y]
            list_name_final = [x for x, y in zip(list_name_final, retain_index) if y]

        else:
            list_name_final =  []
        
        if 'IC' in cv_type:
            return(hyperparams, DALVEN_model, DALVEN_params, rmse_train, rmse_test, yhat_train, yhat_test, IC_result[ind], list_name_final)
        else:
            return(hyperparams, DALVEN_model, DALVEN_params, rmse_train, rmse_test, yhat_train, yhat_test, RMSE_mean[ind], list_name_final)

    elif model_name == 'RNN':
        import timeseries_regression_RNN as RNN
        input_size_x = X.shape[1]

        # Model architecture
        if 'cell_type' not in kwargs:
            kwargs['cell_type'] = ['basic']
        if 'activation' not in kwargs:
            kwargs['activation'] = ['tanh']
        if 'RNN_layers' not in kwargs:
            kwargs['RNN_layers'] = [[input_size_x]]

        # Training parameters
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 1
        if 'epoch_overlap' not in kwargs:
            kwargs['epoch_overlap'] = None
        if 'num_steps' not in kwargs:
            kwargs['num_steps'] = 10
        if 'learning_rate' not in kwargs:
            kwargs['learning_rate'] = 1e-3
        if 'lambda_l2_reg' not in kwargs:
            kwargs['lambda_l2_reg'] = 1e-3
        if 'num_epochs' not in kwargs:
            kwargs['num_epochs'] = 200
        # Dropout parameters
        if 'input_prob' not in kwargs:
            kwargs['input_prob'] = 0.95
        if 'output_prob' not in kwargs:
            kwargs['output_prob'] = 0.95
        if 'state_prob' not in kwargs:
            kwargs['state_prob'] = 0.95
        # Currently we do not support BRNNs, so always keep all neurons during test
        input_prob_test = 1
        output_prob_test = 1
        state_prob_test = 1

        # Early stopping
        if 'val_ratio' not in kwargs and X_val is None:
            kwargs['val_ratio'] = 0.2
        else:
            kwards['val_ratio'] = 0
        if 'max_checks_without_progress' not in kwargs:
            kwargs['max_checks_without_progress'] = 50
        if 'epoch_before_val' not in kwargs:
            kwargs['epoch_before_val'] = 50

        if 'save_location' not in kwargs:
            kwargs['save_location'] = 'RNN_feedback_0'
        if 'plot' not in kwargs:
            kwargs['plot'] = False

        if 'IC' in cv_type: # Information criterion
            IC_result = np.zeros( (len(kwargs['cell_type']), len(kwargs['activation']), len(kwargs['RNN_layers'])) )
            for i in range(len(kwargs['cell_type'])):
                for j in range(len(kwargs['activation'])):
                    for k in range(len(kwargs['RNN_layers'])):
                        _, _, _, (AIC,AICc,BIC), _, _, _ = RNN.timeseries_RNN_feedback_single_train(X, y, X_val, y_val, None, None, kwargs['val_ratio'], kwargs['cell_type'][i],
                                    kwargs['activation'][j], kwargs['RNN_layers'][k], kwargs['batch_size'], kwargs['epoch_overlap'], kwargs['num_steps'], kwargs['learning_rate'],
                                    kwargs['lambda_l2_reg'], kwargs['num_epochs'], kwargs['input_prob'], kwargs['output_prob'], kwargs['state_prob'], input_prob_test,
                                    output_prob_test, state_prob_test, kwargs['max_checks_without_progress'], kwargs['epoch_before_val'], kwargs['save_location'], plot = False)
                        if cv_type == 'AICc':
                            IC_result[i,j,k] = AICc
                        elif cv_type == 'BIC':
                            IC_result[i,j,k] = BIC
                        else:
                            IC_result[i,j,k] = AIC
            # Min IC value (first occurrence)
            ind = np.unravel_index(np.argmin(IC_result, axis=None), IC_result.shape)
        else: # Cross-validation
            RMSE_result = np.empty((len(kwargs['cell_type']), len(kwargs['activation']), len(kwargs['RNN_layers']), K_fold * Nr)) * np.nan
            if kwargs['robust_priority']:
                S = np.empty((len(kwargs['cell_type']), len(kwargs['activation']), len(kwargs['RNN_layers']), K_fold*Nr)) * np.nan

            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, cv_type, K_fold, Nr, group = group)):
                for i in range(len(kwargs['cell_type'])):
                    for j in range(len(kwargs['activation'])):
                        for k in range(len(kwargs['RNN_layers'])):
                            _, _, _, _, _, val_loss, _ = RNN.timeseries_RNN_feedback_single_train(X, y, X_val, y_val, None, None, kwargs['val_ratio'], kwargs['cell_type'][i],
                                    kwargs['activation'][j], kwargs['RNN_layers'][k], kwargs['batch_size'], kwargs['epoch_overlap'], kwargs['num_steps'], kwargs['learning_rate'],
                                    kwargs['lambda_l2_reg'], kwargs['num_epochs'], kwargs['input_prob'], kwargs['output_prob'], kwargs['state_prob'], input_prob_test,
                                    output_prob_test, state_prob_test, kwargs['max_checks_without_progress'], kwargs['epoch_before_val'], kwargs['save_location'], plot = False)
                            RMSE_result[i, j, k, counter] = val_loss
                            if kwargs['robust_priority']:
                                S[i, j, k, counter] = k + i + j # TODO: is this scoring system correct? It ignores the actual values of the paramters, caring only about their positions in the array.

            RMSE_mean = np.nanmean(RMSE_result, axis = 3)
            # Min MSE value (first occurrence)
            ind = np.unravel_index(np.nanargmin(RMSE_mean), RMSE_mean.shape)
            if kwargs['robust_priority']:
                RMSE_std = np.nanstd(RMSE_result, axis = 3)
                RMSE_min = RMSE_mean[ind]
                RMSE_bar = RMSE_min + RMSE_std[ind]
                S_val = np.nansum(S, axis = 3)
                ind = np.nonzero(S_val == np.nanmin(S_val[RMSE_mean < RMSE_bar])) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
                ind = (ind[0][0], ind[1][0], ind[2][0])

        # Hyperparameter setup
        cell_type = kwargs['cell_type'][ind[0]]
        activation = kwargs['activation'][ind[1]]
        RNN_layers = kwargs['RNN_layers'][ind[2]]

        prediction_train, prediction_val, prediction_test, _, train_loss_final, val_loss_final, test_loss_final = RNN.timeseries_RNN_feedback_single_train(X, y, None, None, X_test, y_test,
                kwargs['val_ratio'], cell_type, activation, RNN_layers, kwargs['batch_size'], kwargs['epoch_overlap'], kwargs['num_steps'], kwargs['learning_rate'], kwargs['lambda_l2_reg'],
                kwargs['num_epochs'], kwargs['input_prob'], kwargs['output_prob'], kwargs['state_prob'], input_prob_test, output_prob_test, state_prob_test, kwargs['max_checks_without_progress'],
                kwargs['epoch_before_val'], kwargs['save_location'], kwargs['plot'])

        hyperparams = {}
        hyperparams['cell_type'] = cell_type
        hyperparams['activation'] = activation
        hyperparams['RNN_layers'] = RNN_layers
        hyperparams['training_params'] = {'batch_size': kwargs['batch_size'], 'epoch_overlap': kwargs['epoch_overlap'], 'num_steps': kwargs['num_steps'], 'learning_rate': kwargs['learning_rate'],
                                        'lambda_l2_reg': kwargs['lambda_l2_reg'], 'num_epochs': kwargs['num_epochs']}
        hyperparams['drop_out'] = {'input_prob': kwargs['input_prob'], 'output_prob': kwargs['output_prob'], 'state_prob': kwargs['state_prob']}
        hyperparams['early_stop'] = {'val_ratio': kwargs['val_ratio'], 'max_checks_without_progress': kwargs['max_checks_without_progress'], 'epoch_before_val': kwargs['epoch_before_val']}
        if 'IC' in cv_type:
            hyperparams['IC_optimal'] = IC_result[ind]
        else:
            hyperparams['MSE_val'] = RMSE_mean[ind]
        return(hyperparams, kwargs['save_location'], prediction_train, prediction_val, prediction_test, train_loss_final, val_loss_final, test_loss_final)

@ignore_warnings()
def _LCEN_joblib_fun(X_train, y_train, X_val, y_val, eps, kwargs, all_pos, prod_idx, this_prod, counter = -1):
    """
    A helper function to parallelize LCEN. Shouldn't be called by the user
    """
    degree, l1_ratio, alpha, lag = this_prod
    if (prod_idx == 0 or not (prod_idx+1)%200) and counter >= 0: # CV
        print(f'Beginning run {prod_idx+1:4} of fold {counter+1:3}', end = '\r')
    elif prod_idx == 0 or not (prod_idx+1)%200: # IC -- no folds
        print(f'Beginning run {prod_idx+1:4}', end = '\r')
    _, variable, _, mse, _, _, _, ICs = rm.LCEN_fitting(X_train, y_train, X_val, y_val, alpha, l1_ratio, degree, lag,
                                tol = eps, trans_type = kwargs['trans_type'], LCEN_type = kwargs['model_name'], selection = kwargs['selection'], all_pos = all_pos)
    return mse, np.sum(variable.flatten() != 0), ICs

@ignore_warnings()
def _SPLS_joblib_fun(X_train, y_train, X_val, y_val, eps, alpha_num, kwargs, counter, prod_idx, this_prod):
    """
    A helper function to parallelize SPLS. Shouldn't be called by the user
    """
    scaler_x_train = StandardScaler(with_mean=True, with_std=True)
    scaler_x_train.fit(X_train)
    X_train_scale = scaler_x_train.transform(X_train)
    X_val_scale = scaler_x_train.transform(X_val)
    scaler_y_train = StandardScaler(with_mean=True, with_std=True)
    scaler_y_train.fit(y_train)
    y_train_scale = scaler_y_train.transform(y_train)
    y_val_scale = scaler_y_train.transform(y_val)
    print(f'Beginning run {prod_idx+1:3} of fold {counter+1:3}', end = '\r')
    rmse = np.nan
    var = np.nan
    if this_prod[0] <= X_train.shape[0] - 1:
        _, variable, _, _, _, yhat_val = rm.SPLS_fitting(X_train_scale, y_train_scale, X_val_scale, y_val_scale, eta = this_prod[1], K = this_prod[0])
        rmse = np.sqrt(np.sum((yhat_val.reshape(-1,1) - y_val_scale) ** 2) / y_val_scale.shape[0])
        var = np.sum(variable.flatten() != 0)
    return rmse, var

@ignore_warnings()
def _EN_joblib_fun(X_train, y_train, X_val, y_val, eps, kwargs, prod_idx, this_prod):
    """
    A helper function to parallelize EN. Shouldn't be called by the user
    """
    scaler_x_train = StandardScaler(with_mean=True, with_std=True)
    scaler_x_train.fit(X_train)
    X_train_scale = scaler_x_train.transform(X_train)
    X_val_scale = scaler_x_train.transform(X_val)
    scaler_y_train = StandardScaler(with_mean=True, with_std=True)
    scaler_y_train.fit(y_train)
    y_train_scale = scaler_y_train.transform(y_train)
    y_val_scale = scaler_y_train.transform(y_val)

    l1_ratio = this_prod[0]

    if l1_ratio == 0:
        alpha_list = kwargs['alpha']

        clf = Ridge(alpha=alpha_list[this_prod[1]], fit_intercept=False).fit(X_train_scale, y_train_scale)
        yhat_val = clf.predict(X_val_scale)
        rmse = np.sqrt(
            np.sum((yhat_val.reshape(-1, 1) - y_val_scale) ** 2) / y_val_scale.shape[0])
        var = np.sum(clf.coef_.flatten() != 0)

    else:
        alpha_list = kwargs['alpha']

        _, variable, _, rmse, _, _ = rm.EN_fitting(X_train_scale, y_train_scale, X_val_scale, y_val_scale,
                                                alpha_list[this_prod[1]], l1_ratio, use_cross_entropy=kwargs['use_cross_entropy'])
        var = np.sum(variable.flatten() != 0)

    return rmse, var

@ignore_warnings()
def _XGB_joblib_fun(X_train, y_train, X_val, y_val, kwargs, prod_idx, this_prod):
    """
    A helper function to parallelize EN. Shouldn't be called by the user
    """
    scaler_x_train = StandardScaler(with_mean=True, with_std=True)
    scaler_x_train.fit(X_train)
    X_train_scale = scaler_x_train.transform(X_train)
    X_val_scale = scaler_x_train.transform(X_val)
    scaler_y_train = StandardScaler(with_mean=True, with_std=True)
    scaler_y_train.fit(y_train)
    y_train_scale = scaler_y_train.transform(y_train)
    y_val_scale = scaler_y_train.transform(y_val)
    _, _, mse, _, _ = nro.XGB_fitting(X_train_scale, y_train_scale, X_val_scale, y_val_scale,
                                            this_prod[0], this_prod[1], 0, 100)

    return mse

@ignore_warnings()
def _DALVEN_joblib_fun(X_train, y_train, X_val, y_val, eps, alpha_num, kwargs, counter, prod_idx, this_prod):
    """
    A helper function to parallelize DALVEN. Shouldn't be called by the user
    """
    print(f'Beginning run {prod_idx+1:3} of fold {counter+1:3}', end = '\r')
    if kwargs['model_name'] == 'DALVEN':
        _, variable, _, mse, _, _ , _, _, ICs = rm.DALVEN_fitting(X_train, y_train, X_val, y_val, alpha = this_prod[2], l1_ratio = this_prod[1],
                                    degree = this_prod[0], lag = this_prod[3], tol = eps , alpha_num = alpha_num, cv = True, selection = kwargs['selection'],
                                    select_value = kwargs['select_value'], trans_type = kwargs['trans_type'], use_cross_entropy = kwargs['use_cross_entropy'])
    else:
        _, variable, _, mse, _, _ , _, _, ICs = rm.DALVEN_fitting_full_nonlinear(X_train, y_train, X_val, y_val, alpha = this_prod[2], l1_ratio = this_prod[1],
                                    degree = this_prod[0], lag = this_prod[3], tol = eps , alpha_num = alpha_num, cv = True, selection = kwargs['selection'],
                                    select_value = kwargs['select_value'], trans_type = kwargs['trans_type'], use_cross_entropy = kwargs['use_cross_entropy'])
    return mse, np.sum(variable.flatten() != 0), ICs

@ignore_warnings()
def _ALVEN_joblib_fun(X_train, y_train, X_val, y_val, eps, alpha_num, kwargs, counter, prod_idx, this_prod):
    """
    A helper function to parallelize ALVEN. Shouldn't be called by the user
    """
    print(f'Beginning run {prod_idx+1:3} of fold {counter+1:3}', end = '\r')
    _, variable, _, _, _, yhat_val, _, _, _ = rm.ALVEN_fitting(X_train, y_train, X_val, y_val, alpha = this_prod[2], l1_ratio = this_prod[1],
                                degree = this_prod[0], tol = eps , alpha_num = alpha_num, cv = True, selection = kwargs['selection'],
                                select_value = kwargs['select_value'], trans_type = kwargs['trans_type'], use_cross_entropy = kwargs['use_cross_entropy'])
    rmse = np.sqrt(np.sum((yhat_val - y_val) ** 2) / y_val.shape[0])
    return rmse, np.sum(variable.flatten() != 0)