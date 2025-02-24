from SPA import main_SPA
from pandas import read_excel

dir_datafile = "C:/Users/ChemeGrad2021/PycharmProjects/Formation_feature_design/Features_tsfresh_autoML"
import glob, os
data_name_list = os.listdir(dir_datafile)
if_logtransform = True

for ii in range(len(data_name_list)):
    data_name = data_name_list[ii].replace('.xlsx', '')
    Data = read_excel(dir_datafile + "/" + data_name + ".xlsx", header = None)

    if Data.shape[1] > 1:
        if if_logtransform:
            main_SPA(data_name="Features_tsfresh_autoML/log_" + data_name,
                     main_data=dir_datafile + "/" + data_name + ".xlsx",
                     save_name="SPA_results_autoML/log_" + data_name + ".pkl",
                     interrogation_plot_name = "log_" + data_name,
                     model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                     interpretable=True, robust_priority=False, group_name='group_labels.txt', spectral_data=False,
                     plot_interrogation=True, nested_cv=True, num_outer = 5, K_fold=10, Nr=1, alpha_num=20, degree=[1],
                     log_transform=if_logtransform)
        else:
            main_SPA(data_name="Features_tsfresh_autoML/" + data_name,
                     main_data=dir_datafile + "/" + data_name + ".xlsx",
                     save_name="SPA_results_autoML/" + data_name + ".pkl",
                     interrogation_plot_name=data_name,
                     model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                     interpretable=True, robust_priority = False, group_name='group_labels.txt', spectral_data=False,
                     plot_interrogation=True, nested_cv=True, num_outer = 5,K_fold=10, Nr=1, alpha_num=20, degree=[1], log_transform=if_logtransform)
