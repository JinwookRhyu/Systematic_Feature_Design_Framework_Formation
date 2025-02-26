from SPA import main_SPA
from pandas import read_excel
import glob, os

mode = 'autoML' # 'agnostic' / 'autoML' / 'designed'
if_logtransform = True # True / False

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

elif mode == 'designed':
    dir_datafile = os.path.dirname(os.getcwd()) + "/Features_designed"
    data_name_list = os.listdir(dir_datafile)

    if not os.path.exists(os.path.dirname(os.getcwd()) + "/SPA_results_designed"):
        os.mkdir(os.path.dirname(os.getcwd()) + "/SPA_results_designed")

    data_type = "log_B_Q_V"
    data_name_list = []
    test_data_list = []
    grouplabel_list = []
    for k in range(5):
        data_name_list.append(data_type + "_train_Outer_" + str(k+1) + ".xlsx")
        test_data_list.append(data_type + "_test_Outer_" + str(k + 1) + ".xlsx")
        grouplabel_list.append(data_type + "_grouplabel_Outer_" + str(k + 1) + ".xlsx")
    
    if_logtransform = False # log already applied in the designed features
    
    for ii in range(len(data_name_list)):
        data_name = data_name_list[ii].replace('.xlsx', '')
        test_data = test_data_list[ii]
        grouplabel = grouplabel_list[ii]
        Data = read_excel(dir_datafile + "/" + data_name + ".xlsx", header = None)
    
        if Data.shape[1] > 1:
            if if_logtransform:
                main_SPA(data_name=os.path.dirname(os.getcwd()) + "/Features_designed/log_" + data_name,
                         main_data=dir_datafile + "/" + data_name + ".xlsx",
                         test_data = os.path.dirname(os.getcwd()) + "/Features_designed/" + test_data,
                         save_name=os.path.dirname(os.getcwd()) + "/SPA_results_designed/log_" + data_name + ".pkl",
                         interrogation_plot_name = "log_" + data_name,
                         model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                         interpretable=True, robust_priority=False, group_name=os.path.dirname(os.getcwd()) + "/Features_designed/" + grouplabel, spectral_data=False,
                         plot_interrogation=True, nested_cv=False, num_outer = 5, K_fold=10, Nr=1, alpha_num=20, degree=[1],
                         log_transform=if_logtransform)
            else:
                main_SPA(data_name=os.path.dirname(os.getcwd()) + "/Features_designed/" + data_name,
                         main_data=dir_datafile + "/" + data_name + ".xlsx",
                         test_data=os.path.dirname(os.getcwd()) + "/Features_designed/" + test_data,
                         save_name=os.path.dirname(os.getcwd()) + "/SPA_results_designed/" + data_name + ".pkl",
                         interrogation_plot_name=data_name,
                         model_name=['XGB', 'RF', 'SVR', 'ALVEN', 'LCEN', 'EN', 'RR', 'PLS'],
                         interpretable=True, robust_priority = False, group_name=os.path.dirname(os.getcwd()) + "/Features_designed/" + grouplabel, spectral_data=False,
                         plot_interrogation=True, nested_cv=False, num_outer = 5,K_fold=10, Nr=1, alpha_num=20, degree=[1], log_transform=if_logtransform)
