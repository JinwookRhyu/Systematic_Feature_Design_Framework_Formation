from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm

dir_pklfile = "C:/Users/Jinwook/PyCharm_projects/Formation_feature/SPA_results_params"
import glob, os
data_name_list = os.listdir(dir_pklfile)

#data_name_list = ["protocol_params_reduced_ALVEN_deg1.pkl"]

model_name_list_all = ['SVR', 'RF', 'EN', 'RR', 'PLS', 'SPLS', 'ALVEN1', 'LCEN1', 'ALVEN2', 'LCEN2', 'ALVEN3', 'LCEN3', 'XGB']

df_data = [[0]*9 for i in range(len(data_name_list) * len(model_name_list_all))]

ind_model = 0

for ii in range(len(data_name_list)):

    data_name = data_name_list[ii].replace('.pkl', '')

    with open("SPA_results_params/" + data_name + ".pkl", 'rb') as file:
        history = pickle.load(file)

    model_name_list = [i for i in history]

    for model_name in model_name_list:

        y_train_list = []
        yhat_train_list = []
        y_test_list = []
        yhat_test_list = []
        train_nest_rmse = []
        test_nest_rmse = []
        train_nest_mape = []
        test_nest_mape = []

        num_nest = len(history[model_name])

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

            train_nest_rmse = np.append(train_nest_rmse,
                                        np.sqrt(np.sum((yhat_train - y_train) ** 2) / y_train.shape[0]))
            test_nest_rmse = np.append(test_nest_rmse, np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0]))
            train_nest_mape = np.append(train_nest_mape,
                                        np.sum(np.divide(np.abs(yhat_train - y_train), y_train) * 100) / y_train.shape[
                                            0])
            test_nest_mape = np.append(test_nest_mape,
                                       np.sum(np.divide(np.abs(yhat_test - y_test), y_test) * 100) / y_test.shape[0])
       
        if 'reduced' in data_name:
            df_data[ind_model][0] = True
            if 'log' in data_name:
                df_data[ind_model][1] = True
                items = data_name.split('_')
            else:
                df_data[ind_model][1] = False
                items = data_name.split('_')
        else:
            df_data[ind_model][0] = False
            if 'log' in data_name:
                df_data[ind_model][1] = True
                items = data_name.split('_')
            else:
                df_data[ind_model][1] = False
                items = data_name.split('_')

        df_data[ind_model][2] = model_name
        df_data[ind_model][3] = np.median(train_nest_rmse)
        df_data[ind_model][4] = np.median(test_nest_rmse)
        df_data[ind_model][5] = np.max(test_nest_rmse)
        df_data[ind_model][6] = np.median(train_nest_mape)
        df_data[ind_model][7] = np.median(test_nest_mape)
        df_data[ind_model][8] = np.max(test_nest_mape)
        ind_model += 1

df = pd.DataFrame(df_data, columns=['if_reduced', 'if_log', 'model', 'med_train_rmse', 'med_test_rmse', 'max_test_rmse', 'med_train_mape', 'med_test_mape', 'max_test_mape'])
df.to_csv('params_results.csv')
