from SPA import main_SPA
from pandas import read_excel

dir_datafile = "C:/Users/ChemeGrad2021/PycharmProjects/Formation_feature_design/designed_features"
import glob, os

data_type = "log_B_Q_V"
data_name_list = []
test_data_list = []
grouplabel_list = []
for k in range(5):
    data_name_list.append(data_type + "_train_Outer_" + str(k+1) + ".xlsx")
    test_data_list.append(data_type + "_test_Outer_" + str(k + 1) + ".xlsx")
    grouplabel_list.append(data_type + "_grouplabel_Outer_" + str(k + 1) + ".xlsx")

if_logtransform = False

for ii in range(len(data_name_list)):
    data_name = data_name_list[ii].replace('.xlsx', '')
    test_data = test_data_list[ii]
    grouplabel = grouplabel_list[ii]
    Data = read_excel(dir_datafile + "/" + data_name + ".xlsx", header = None)

    if Data.shape[1] > 1:
        if if_logtransform:
            main_SPA(data_name="designed_features/log_" + data_name,
                     main_data=dir_datafile + "/" + data_name + ".xlsx",
                     test_data = "designed_features/" + test_data,
                     save_name="SPA_results_designed/log_" + data_name + ".pkl",
                     interrogation_plot_name = "log_" + data_name,
                     model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                     interpretable=True, robust_priority=False, group_name="designed_features/" + grouplabel, spectral_data=False,
                     plot_interrogation=True, nested_cv=False, num_outer = 5, K_fold=10, Nr=1, alpha_num=20, degree=[1],
                     log_transform=if_logtransform)
        else:
            main_SPA(data_name="designed_features/" + data_name,
                     main_data=dir_datafile + "/" + data_name + ".xlsx",
                     test_data="designed_features/" + test_data,
                     save_name="SPA_results_designed/" + data_name + ".pkl",
                     interrogation_plot_name=data_name,
                     model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                     interpretable=True, robust_priority = False, group_name="designed_features/" + grouplabel, spectral_data=False,
                     plot_interrogation=True, nested_cv=False, num_outer = 5,K_fold=10, Nr=1, alpha_num=20, degree=[1], log_transform=if_logtransform)
