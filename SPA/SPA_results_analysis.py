from matplotlib.pyplot import cm
import numpy as np
from pandas import read_excel, read_csv
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm
import regression_models as rm
from sklearn.preprocessing import StandardScaler
import shap
import sys

file_name = "protocol_params"
model_type = "params" # "params" for agnostic model, "autoML" for autoML model, "designed" for model with designed features
model_name_list = ["SVR"]

if model_type == "params":
    SPA_results_folder = "SPA_results_params/"
    SPA_data_folder = "SPA_data_params/"
elif model_type == "autoML":
    SPA_results_folder = "SPA_results_autoML/"
    SPA_data_folder = "SPA_data_autoML/"
elif model_type == "designed":
    SPA_results_folder = "SPA_results_designed/"
    SPA_data_folder = "SPA_data_designed/"
else:
    sys.exit("Wrong model type. Please choose either params / autoML / designed")

with open(SPA_results_folder + file_name + ".pkl", 'rb') as file:
    history = pickle.load(file)

for separator in (' ', ',', '\t', ';'):  # Testing random separators
    my_file = read_csv('group_labels.txt', header=None, sep=separator).values
    if my_file.shape[-1] > 1:  # We likely found the separator
        break
group = my_file.flatten()

group_list = [g for g in range(np.max(group))]
np.random.RandomState(42).shuffle(group_list)
split = np.array_split(group_list, 5)

Data_original = read_excel(SPA_data_folder + file_name + ".xlsx", header = None).values
if "tsfresh" in file_name:
    X_original = Data_original[:, 1:]
    y_original = Data_original[:, 0].reshape(-1, 1)
else:
    X_original = Data_original[:, :-1]
    y_original = Data_original[:, -1].reshape(-1, 1)

Data_name = "Features/" + file_name + ".xlsx"  # input("Name of your data file (Should have dimension N x (m+1), last column for predicted variable) ")
Data = pd.read_excel(Data_name, header=None)
Data = np.array(Data)

threshold = 1

if 'tsfresh' in file_name:
    Feature_list = Data[0, 2:]
else:
    Feature_list = Data[0, 1:]

df = pd.DataFrame()
for model_name in model_name_list:
    for i in range(len(history[model_name])):
        train = [element for element in range(len(group)) if group[element] not in split[i]]
        test = [element for element in range(len(group)) if group[element] in split[i]]

        X = X_original[train]
        X_test = X_original[test]
        y = y_original[train]
        scaler_x = StandardScaler(with_mean=True, with_std=True)
        scaler_x.fit(X)
        X_scale = scaler_x.transform(X)
        scaler_y = StandardScaler(with_mean=True, with_std=True)
        scaler_y.fit(y)
        y_scale = scaler_y.transform(y)


        if "LCEN" in model_name:
            all_pos = np.all(X >= 0, axis=0) & np.all(X_test >= 0, axis=0)
            X, _, label_name = rm._feature_trans(X, degree=history[model_name][i]['model_hyper']['degree'],
                                                 trans_type='all', all_pos=all_pos)
            Feature_list_new = []
            param_value_new = []

            y = history[model_name][i]['y_train']

            for k in range(len(history[model_name][i]['model_params'])):
                ID = history[model_name][i]['model_params'][k][0]
                col_ind = np.where(label_name == ID)[0][0]
                for l in reversed(range(len(Feature_list))):
                    ID = ID.replace("x" + str(l), Feature_list[l])
                Feature_list_new = np.append(Feature_list_new, ID)
                param_value_new = np.append(param_value_new,
                                            history[model_name][i]['model_params'][k][1] / y.std() * X[:, col_ind].std(
                                                axis=0))

            Feature_importance = np.array([Feature_list_new, param_value_new])
        elif "ALVEN" in model_name:
            all_pos = np.all(X >= 0, axis=0) & np.all(X_test >= 0, axis=0)
            X, _, label_name = rm._feature_trans(X, degree=history[model_name][i]['model_hyper']['degree'],
                                                 trans_type='all', all_pos=all_pos)
            Feature_list_new = []
            param_value_new = []

            y = history[model_name][i]['y_train']

            for k in range(len(history[model_name][i]['final_list'])):
                ID = history[model_name][i]['final_list'][k]
                for l in reversed(range(len(Feature_list))):
                    ID = ID.replace("x" + str(l + 1), Feature_list[l])
                Feature_list_new = np.append(Feature_list_new, ID)

            Feature_importance = np.array([Feature_list_new, history[model_name][i]['model_params'].flatten()])
        elif model_name == 'EN':
            Feature_list_new = Feature_list
            Feature_importance = np.array([Feature_list_new, history[model_name][i]['model_params'].flatten()])
        else:
            X_scale = pd.DataFrame(X_scale, columns = Feature_list)
            explainer = shap.Explainer(history[model_name][i]['final_model'].predict, X_scale)
            shap_values = explainer(X_scale)
            shap.plots.beeswarm(shap_values, plot_size=(10, 6))

        df_new = pd.DataFrame(Feature_importance).T
        df_new = df_new.rename(columns={0: 'Features', 1: "Values_" + str(i)})
        df_new = df_new.set_index('Features')
        df = pd.concat([df, df_new], axis=1)

    df_nonan = df.fillna(0)
    appearance = np.sum(df_nonan.to_numpy().astype(float) != 0, axis=1)
    num_features = np.sum(df_nonan.to_numpy().astype(float) != 0, axis=0)
    df['Appearance'] = appearance

    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df = df.sort_values(by=['Appearance'], ascending=False)
