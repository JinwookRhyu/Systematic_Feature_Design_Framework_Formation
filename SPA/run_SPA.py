from SPA import main_SPA
from pandas import read_excel
import glob, os

mode = 'autoML' # agnostic or autoML
if_logtransform = True

if mode == 'agnostic':
    dir_datafile = os.path.dirname(os.getcwd()) + "/Features_agnostic"

    if not os.path.exists(os.path.dirname(os.getcwd()) + "/SPA_results_agnostic"):
        os.mkdir(os.path.dirname(os.getcwd()) + "/SPA_results_agnostic")

    data_name_list = os.listdir(dir_datafile)
    
    for ii in range(len(data_name_list)):
        data_name = data_name_list[ii].replace('.xlsx', '')
        Data = read_excel(dir_datafile + "/" + data_name + ".xlsx", header = None)
    
        if Data.shape[1] > 1:
            if if_logtransform:
                main_SPA(data_name=os.path.dirname(os.getcwd()) + "/Features_agnostic/log_" + data_name,
                         main_data=dir_datafile + "/" + data_name + ".xlsx",
                         save_name=os.path.dirname(os.getcwd()) + "/SPA_results_agnostic/log_" + data_name + ".pkl",
                         interrogation_plot_name = "log_" + data_name,
                         model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                         interpretable=True, robust_priority=False, group_name='group_labels.txt', spectral_data=False,
                         plot_interrogation=True, nested_cv=True, num_outer = 5, K_fold=10, Nr=1, alpha_num=20, degree=[3],
                         log_transform=if_logtransform)
            else:
                main_SPA(data_name=os.path.dirname(os.getcwd()) + "/Features_agnostic/" + data_name,
                         main_data=dir_datafile + "/" + data_name + ".xlsx",
                         save_name=os.path.dirname(os.getcwd()) + "/SPA_results_agnostic/" + data_name + ".pkl",
                         interrogation_plot_name=data_name,
                         model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                         interpretable=True, robust_priority = False, group_name='group_labels.txt', spectral_data=False,
                         plot_interrogation=True, nested_cv=True, num_outer = 5,K_fold=10, Nr=1, alpha_num=20, degree=[3], log_transform=if_logtransform)

elif mode == 'autoML':
    dir_datafile = os.path.dirname(os.getcwd()) + "/Features_tsfresh_autoML"
    # Navigate to the directory with Features_tsfresh_autoML folder. This SPA code also can be used for agnostic models when replacing "Features_tsfresh_autoML" with "Features_agnostic" and also from "SPA_results_autoML" with "SPA_results_agnostic"
    data_name_list = os.listdir(dir_datafile)

    if not os.path.exists(os.path.dirname(os.getcwd()) + "/SPA_results_autoML"):
        os.mkdir(os.path.dirname(os.getcwd()) + "/SPA_results_autoML")
    
    for ii in range(len(data_name_list)):
        data_name = data_name_list[ii].replace('.xlsx', '')
        Data = read_excel(dir_datafile + "/" + data_name + ".xlsx", header = None)
    
        if Data.shape[1] > 1:
            if if_logtransform:
                main_SPA(data_name=os.path.dirname(os.getcwd()) + "/Features_tsfresh_autoML/log_" + data_name,
                         main_data=dir_datafile + "/" + data_name + ".xlsx",
                         save_name=os.path.dirname(os.getcwd()) + "/SPA_results_autoML/log_" + data_name + ".pkl",
                         interrogation_plot_name = "log_" + data_name,
                         model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                         interpretable=True, robust_priority=False, group_name='group_labels.txt', spectral_data=False,
                         plot_interrogation=True, nested_cv=True, num_outer = 5, K_fold=10, Nr=1, alpha_num=20, degree=[1],
                         log_transform=if_logtransform)
            else:
                main_SPA(data_name=os.path.dirname(os.getcwd()) + "/Features_tsfresh_autoML/" + data_name,
                         main_data=dir_datafile + "/" + data_name + ".xlsx",
                         save_name=os.path.dirname(os.getcwd()) + "/SPA_results_autoML/" + data_name + ".pkl",
                         interrogation_plot_name=data_name,
                         model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                         interpretable=True, robust_priority = False, group_name='group_labels.txt', spectral_data=False,
                         plot_interrogation=True, nested_cv=True, num_outer = 5,K_fold=10, Nr=1, alpha_num=20, degree=[1], log_transform=if_logtransform)
