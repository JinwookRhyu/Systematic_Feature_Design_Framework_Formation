from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm

dir_pklfile = "C:/Users/Jinwook/PyCharm_projects/Formation_feature/SPA_results_autoML"
import glob, os
data_name_list = os.listdir(dir_pklfile)
data_name_list = ["log_B_V_t_features_univariate_tsfresh_3.5", "log_B_V_t_features_univariate_tsfresh_4.5"]

for ii in range(len(data_name_list)):

    data_name = data_name_list[ii].replace('.pkl', '')

    with open("SPA_results_autoML/" + data_name + ".pkl", 'rb') as file:
        history = pickle.load(file)

    model_name_list = ['EN', 'RF', 'SVR', 'ALVEN', 'LCEN', 'XGB']
    model_name_display_list = model_name_list

    rmse_train = []
    rmse_test = []
    mape_train = []
    mape_test = []
    model_names_display = []
    num_nest_display = []
    for model_name in model_name_list:

        save_name = "SPA_plots_autoML/" + data_name + "_Fit_" + model_name + ".png"

        y_train_list = []
        yhat_train_list = []
        y_test_list = []
        yhat_test_list = []
        train_nest_rmse = []
        test_nest_rmse = []
        train_nest_mape = []
        test_nest_mape = []

        fig, ax = plt.subplots(1, 4, figsize=(12, 4))

        num_nest = len(history[model_name])
        if num_nest > 0:
            color = iter(cm.rainbow(np.linspace(0, 1, num_nest)))

            for i in list(history[model_name].keys()):
                c = next(color)

                y_train = history[model_name][i]['y_train']
                yhat_train = history[model_name][i]['yhat_train']
                y_test = history[model_name][i]['y_test']
                yhat_test = history[model_name][i]['yhat_test']

                if "log" in data_name:
                    y_train = np.exp(y_train)
                    yhat_train = np.exp(yhat_train)
                    y_test = np.exp(y_test)
                    yhat_test = np.exp(yhat_test)

                y_train_list = np.append(y_train_list, y_train.flatten())
                yhat_train_list = np.append(yhat_train_list, yhat_train.flatten())
                y_test_list = np.append(y_test_list, y_test.flatten())
                yhat_test_list = np.append(yhat_test_list, yhat_test.flatten())

                train_nest_rmse = np.append(train_nest_rmse,
                                            np.sqrt(np.sum((yhat_train - y_train) ** 2) / y_train.shape[0]))
                test_nest_rmse = np.append(test_nest_rmse, np.sqrt(np.sum((yhat_test - y_test) ** 2) / y_test.shape[0]))
                train_nest_mape = np.append(train_nest_mape, np.sum(np.divide(np.abs(yhat_train - y_train), y_train) * 100) / y_train.shape[0])
                test_nest_mape = np.append(test_nest_mape, np.sum(np.divide(np.abs(yhat_test - y_test), y_test) * 100) / y_test.shape[0])

                '''Basic Residual Plot'''

                # axm = plt.axes()
                # axm.set_facecolor("white")
                # plt.plot(ytrain,ytrain_hat,'*',label = 'Training data')
                ax[0].plot(y_train, yhat_train, '*', c=c)
                ax[1].plot(y_test, yhat_test, '*', c=c)
                # Legend position varies depending on plot (change here)

                df = pd.DataFrame(abs(y_train - yhat_train).flatten(), columns=['Error'])
                df['Model Type'] = np.repeat(model_name, len(y_train))
                df['Split Type'] = np.repeat('Train', len(y_train))
                df1 = pd.DataFrame(abs(y_test - yhat_test).flatten(), columns=['Error'])
                df1['Model Type'] = np.repeat(model_name, len(y_test))
                df1['Split Type'] = np.repeat('Test', len(y_test))

                df = pd.concat([df, df1], ignore_index=True)

                sns.violinplot(
                   data=df,
                   x='Model Type', y='Error', cut = 0, fill=False, gap=.2, palette= [c, c],
                   hue='Split Type', split=True, density_norm='area', inner='quart', legend=False, ax=ax[2])
                ax[2].collections[2*i].set_edgecolor(c)

                df2 = pd.DataFrame(100 * np.divide(abs(y_train - yhat_train), y_train).flatten(),
                                   columns=['Relative Error'])
                df2['Model Type'] = np.repeat(model_name, len(y_train))
                df2['Split Type'] = np.repeat('Train', len(y_train))

                df3 = pd.DataFrame(100 * np.divide(abs(y_test - yhat_test), y_test).flatten(),
                                   columns=['Relative Error'])
                df3['Model Type'] = np.repeat(model_name, len(y_test))
                df3['Split Type'] = np.repeat('Test', len(y_test))

                df2 = pd.concat([df2, df3], ignore_index=True)

                sns.violinplot(
                    data=df2,
                    x='Model Type', y='Relative Error', cut = 0, fill=False, gap=.2, palette= [c, c],
                    hue='Split Type', split=True, density_norm='area', inner='quart', legend=False, ax=ax[3])
                ax[3].collections[2 * i].set_edgecolor(c)
            # ax.legend(loc='lower right', fontsize = 10)

            sm.qqline(ax=ax[0], line='45', fmt='k--')
            sm.qqline(ax=ax[1], line='45', fmt='k--')
            ax[0].set_ylabel('fitted y', fontsize=14)
            ax[0].set_xlabel('y', fontsize=14)
            ax[0].axis('scaled')
            ax[0].set_title("\nmed_RMSE_train = " + str(np.round(np.median(train_nest_rmse) * 100) / 100) + "\nmax_RMSE_train = " + str(np.round(np.max(train_nest_rmse) * 100) / 100) + "\nmed_MAPE_train = " + str(np.round(np.median(train_nest_mape) * 100) / 100) + "\nmax_MAPE_train = " + str(np.round(np.max(train_nest_mape) * 100) / 100))
            ax[1].set_ylabel('fitted y', fontsize=14)
            ax[1].set_xlabel('y', fontsize=14)
            ax[1].axis('scaled')
            ax[1].set_title("med_RMSE_test = " + str(np.round(np.median(test_nest_rmse) * 100) / 100) + "\nmax_RMSE_test = " + str(np.round(np.max(test_nest_rmse) * 100) / 100) + "\nmed_MAPE_test = " + str(np.round(np.median(test_nest_mape) * 100) / 100) + "\nmax_MAPE_test = " + str(np.round(np.max(test_nest_mape) * 100) / 100))
            # plt.show()

            color = iter(cm.rainbow(np.linspace(0, 1, num_nest)))

            for i in list(history[model_name].keys()):
                c = next(color)
                ax[2].scatter(x=-0.04, y=train_nest_rmse[i], color=c, edgecolors='k', s=100)
                ax[2].scatter(x=0.04, y=test_nest_rmse[i], color=c, edgecolors='k', s=100)
                ax[3].scatter(x=-0.04, y=train_nest_mape[i], color=c, edgecolors='k', s=100)
                ax[3].scatter(x=0.04, y=test_nest_mape[i], color=c, edgecolors='k', s=100)


            ax[2].set_title('RMSE from nested CV')
            ax[3].set_title('MAPE from nested CV')

            plt.tight_layout()

            plt.savefig(save_name, dpi=600, bbox_inches='tight')

            rmse_train = np.append(rmse_train, train_nest_rmse)
            rmse_test = np.append(rmse_test, test_nest_rmse)
            mape_train = np.append(mape_train, train_nest_mape)
            mape_test = np.append(mape_test, test_nest_mape)

            model_names_display += [model_name]
            num_nest_display = np.append(num_nest_display, num_nest)

    num_nest_display = num_nest_display.astype(int)
    df4 = pd.DataFrame(rmse_train.flatten(), columns=['RMSE'])
    df4['Model Type'] = np.repeat(model_names_display, num_nest_display)
    df4['Split Type'] = np.repeat('Train', np.sum(num_nest_display))
    df5 = pd.DataFrame(rmse_test.flatten(), columns=['RMSE'])
    df5['Model Type'] = np.repeat(model_names_display, num_nest_display)
    df5['Split Type'] = np.repeat('Test', np.sum(num_nest_display))

    df4 = pd.concat([df4, df5], ignore_index=True)

    plt.figure()
    ax = sns.violinplot(
        data=df4,
        x='Model Type', y='RMSE',
        hue='Split Type', split=True, palette="colorblind", scale='count', inner='stick')
    ax.set_title('RMSE distribution using nested CV')
    plt.savefig(
        "SPA_plots_autoML/" + data_name + "_Violin_RMSE.png")

    df6 = pd.DataFrame(mape_train.flatten(), columns=['MAPE'])
    df6['Model Type'] = np.repeat(model_names_display, num_nest_display)
    df6['Split Type'] = np.repeat('Train', np.sum(num_nest_display))
    df7 = pd.DataFrame(mape_test.flatten(), columns=['MAPE'])
    df7['Model Type'] = np.repeat(model_names_display, num_nest_display)
    df7['Split Type'] = np.repeat('Test', np.sum(num_nest_display))

    df6 = pd.concat([df6, df7], ignore_index=True)

    plt.figure()
    ax = sns.violinplot(
        data=df6,
        x='Model Type', y='MAPE',
        hue='Split Type', split=True, palette="colorblind", scale='count', inner='stick')
    ax.set_title('MAPE distribution using nested CV')
    plt.savefig(
        "SPA_plots_autoML/" + data_name + "_Violin_MAPE.png")