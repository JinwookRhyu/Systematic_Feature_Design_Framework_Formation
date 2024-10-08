from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm

dir_pklfile = "C:/Users/ChemeGrad2021/PycharmProjects/Formation_feature_design/SPA_results_designed"
import glob, os
#data_name_list = os.listdir(dir_pklfile)
data_type = "log_B_Q_V"
# history = {}
# for k in range(5):
#     with open("SPA_results_designed/" + data_type + "_train_Outer_" + str(k+1) + ".pkl", 'rb') as file:
#         history_outer = pickle.load(file)
#     with open("SPA_results_designed/" + data_type + "_train_Outer_" + str(k+1) + "_deg2.pkl", 'rb') as file:
#         history_outer_deg2 = pickle.load(file)
#     with open("SPA_results_designed/" + data_type + "_train_Outer_" + str(k+1) + "_deg3.pkl", 'rb') as file:
#         history_outer_deg3 = pickle.load(file)
#     keylist = list(history_outer.keys())
#     keylist.append('ALVEN2')
#     keylist.append('ALVEN3')
#     keylist.append('LCEN2')
#     keylist.append('LCEN3')
#     for key in keylist:
#         if k == 0:
#             history[key] = {}
#         if key in history_outer.keys():
#             history[key][k] = history_outer[key][0]
#     history['ALVEN2'][k] = history_outer_deg2['ALVEN'][0]
#     history['LCEN2'][k] = history_outer_deg2['LCEN'][0]
#     history['ALVEN3'][k] = history_outer_deg3['ALVEN'][0]
#     history['LCEN3'][k] = history_outer_deg3['LCEN'][0]
#
#
# with open("SPA_results_designed/" + data_type + "_designed.pkl", 'wb') as f:
#     pickle.dump(history, f)

data_name = data_type + "_designed"

with open("SPA_results_designed/" + data_name + ".pkl", 'rb') as file:
    history = pickle.load(file)

model_name_list_all = ['SVR', 'RF', 'EN', 'RR', 'PLS', 'SPLS', 'LCEN', 'ALVEN', 'XGB', 'LCEN2', 'LCEN3', 'ALVEN2', 'ALVEN3']

df_data = [[0]*8 for i in range(len(model_name_list_all))]

ind_model = 0


with open("SPA_results_designed/" + data_name + ".pkl", 'rb') as file:
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

    # ax.legend(loc='lower right', fontsize = 10)

    df_data[ind_model][0] = data_name
    df_data[ind_model][1] = model_name
    df_data[ind_model][2] = np.median(train_nest_rmse)
    df_data[ind_model][3] = np.median(test_nest_rmse)
    df_data[ind_model][4] = np.max(test_nest_rmse)
    df_data[ind_model][5] = np.median(train_nest_mape)
    df_data[ind_model][6] = np.median(test_nest_mape)
    df_data[ind_model][7] = np.max(test_nest_mape)
    ind_model += 1

df = pd.DataFrame(df_data, columns=['data', 'model', 'med_train_rmse', 'med_test_rmse', 'max_test_rmse', 'med_train_mape', 'med_test_mape', 'max_test_mape'])
df.to_csv('designed_results.csv')
