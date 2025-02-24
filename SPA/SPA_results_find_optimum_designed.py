from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm

dir_pklfile = "C:/Users/Jinwook/PyCharm_projects/Formation_feature/SPA_results_designed"
import glob, os
data_name_list = os.listdir(dir_pklfile)

data_name_list = ["log_B_Q_V_designed.pkl"]

model_name_list_all = ['SVR', 'RF', 'EN', 'RR', 'PLS', 'SPLS', 'LCEN', 'ALVEN', 'LCEN2', 'ALVEN2', 'LCEN3', 'ALVEN3', 'XGB']

df_data = [[0]*38 for i in range(len(data_name_list) * len(model_name_list_all))]

ind_model = 0

for ii in range(len(data_name_list)):

    data_name = data_name_list[ii].replace('.pkl', '')

    with open("SPA_results_designed/" + data_name + ".pkl", 'rb') as file:
        history = pickle.load(file)

    model_name_list = [i for i in history]

    for model_name in model_name_list:

        num_nest = len(history[model_name])

        y_train_list = []
        yhat_train_list = []
        y_test_list = []
        yhat_test_list = []
        train_nest_rmse = np.full((5,), np.nan)
        test_nest_rmse = np.full((5,), np.nan)
        train_nest_mape = np.full((5,), np.nan)
        test_nest_mape = np.full((5,), np.nan)

        for i in range(num_nest):
            if i in history[model_name]:
                y_train = history[model_name][i]['y_train']
                yhat_train = history[model_name][i]['yhat_train']
                y_test = history[model_name][i]['y_test']
                yhat_test = history[model_name][i]['yhat_test']

                if "log" in data_name:
                    y_train = np.exp(y_train)
                    yhat_train = np.exp(yhat_train)
                    y_test = np.exp(y_test)
                    yhat_test = np.exp(np.clip(yhat_test, -np.inf, 300))

                y_train_list = np.append(y_train_list, y_train.flatten())
                yhat_train_list = np.append(yhat_train_list, yhat_train.flatten())
                y_test_list = np.append(y_test_list, y_test.flatten())
                yhat_test_list = np.append(yhat_test_list, yhat_test.flatten())

            train_nest_rmse[i] = np.sqrt(np.sum((yhat_train - y_train) ** 2) / y_train.shape[0])
            test_nest_rmse[i] = np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0])
            train_nest_mape[i] = np.sum(np.divide(np.abs(yhat_train - y_train), y_train) * 100) / y_train.shape[0]
            test_nest_mape[i] = np.sum(np.divide(np.abs(yhat_test - y_test), y_test) * 100) / y_test.shape[0]

        # ax.legend(loc='lower right', fontsize = 10)
        train_nest_rmse_pair = np.full((int(0.5 * num_nest * (num_nest + 1)),), np.nan)
        test_nest_rmse_pair = np.full((int(0.5 * num_nest * (num_nest + 1)),), np.nan)
        train_nest_mape_pair = np.full((int(0.5 * num_nest * (num_nest + 1)),), np.nan)
        test_nest_mape_pair = np.full((int(0.5 * num_nest * (num_nest + 1)),), np.nan)
        k_pair = 0
        for iii in range(num_nest):
            for jjj in range(iii + 1):
                train_nest_rmse_pair[k_pair] = 0.5 * (train_nest_rmse[iii] + train_nest_rmse[jjj])
                test_nest_rmse_pair[k_pair] = 0.5 * (test_nest_rmse[iii] + test_nest_rmse[jjj])
                train_nest_mape_pair[k_pair] = 0.5 * (train_nest_mape[iii] + train_nest_mape[jjj])
                test_nest_mape_pair[k_pair] = 0.5 * (test_nest_mape[iii] + test_nest_mape[jjj])
                k_pair = k_pair + 1

        df_data[ind_model][0] = data_name
        df_data[ind_model][1] = model_name
        df_data[ind_model][2:7] = train_nest_rmse
        df_data[ind_model][7:12] = test_nest_rmse
        df_data[ind_model][12:17] = train_nest_mape
        df_data[ind_model][17:22] = test_nest_mape
        df_data[ind_model][22] = np.nanmean(train_nest_rmse)
        df_data[ind_model][23] = np.nanmean(test_nest_rmse)
        df_data[ind_model][24] = np.nanmean(train_nest_mape)
        df_data[ind_model][25] = np.nanmean(test_nest_mape)
        df_data[ind_model][26] = np.nanmedian(train_nest_rmse)
        df_data[ind_model][27] = np.nanmedian(test_nest_rmse)
        df_data[ind_model][28] = np.nanmedian(train_nest_mape)
        df_data[ind_model][29] = np.nanmedian(test_nest_mape)
        df_data[ind_model][30] = np.nanmax(train_nest_rmse)
        df_data[ind_model][31] = np.nanmax(test_nest_rmse)
        df_data[ind_model][32] = np.nanmax(train_nest_mape)
        df_data[ind_model][33] = np.nanmax(test_nest_mape)
        df_data[ind_model][34] = np.nanmedian(train_nest_rmse_pair)
        df_data[ind_model][35] = np.nanmedian(test_nest_rmse_pair)
        df_data[ind_model][36] = np.nanmedian(train_nest_mape_pair)
        df_data[ind_model][37] = np.nanmedian(test_nest_mape_pair)
        ind_model += 1

df = pd.DataFrame(df_data, columns=['data', 'model', 'train_rmse_1', 'train_rmse_2', 'train_rmse_3', 'train_rmse_4', 'train_rmse_5', 'test_rmse_1', 'test_rmse_2', 'test_rmse_3', 'test_rmse_4', 'test_rmse_5', 'train_mape_1', 'train_mape_2', 'train_mape_3', 'train_mape_4', 'train_mape_5', 'test_mape_1', 'test_mape_2', 'test_mape_3', 'test_mape_4', 'test_mape_5', 'mean_train_rmse', 'mean_test_rmse', 'mean_train_mape', 'mean_test_mape', 'med_train_rmse', 'med_test_rmse', 'med_train_mape', 'med_test_mape', 'max_train_rmse', 'max_test_rmse', 'max_train_mape', 'max_test_mape', 'HL_train_rmse', 'HL_test_rmse', 'HL_train_mape', 'HL_test_mape'])
df.to_csv('designed_results.csv')
