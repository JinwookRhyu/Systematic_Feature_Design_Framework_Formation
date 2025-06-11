close all; 
clear all; clc;

%% Settings
data_type = "B_Q_V"; % Data type
legend_position = 'northwest'; % legend location
plot_ylim = [-1.2, 1.2]; % ylim for plot
if_logtransform = true; % Whether to log-transform the data or not
nfolds_outer = 5; % Number of outer loops
nfolds_inner = 5; % Number of inner loops
num_lambda_show = 10; % Number of lamdas to plot
if_savematrix = true; % Whether to save the dataset of selected features

% Robustness and interpretability thresholds as in Table S2
threshold_dtw_ratio_others = 0.7;
threshold_path_length = 5;

merge_threshold = 0.01;
corr_threshold = 0.4;
p_close = 0.01;

% Minor settings for the plots 
fontsize = 20;
ticklabelsize = 15;
cmap = [0.5, 0, 1; 0, 0.71, 0.923; 0.5, 1, 0.705; 1, 0.7, 0.3784; 1, 0, 0];
max_lambda = 1e3;

% Check if dataset is log-transformed
if if_logtransform
    filename = "log_" + data_type + "_metrics.mat";
else
    filename = data_type + "_metrics.mat";
end

X_raw = xlsread("data_formation/" + data_type + "_all.xlsx");
X_raw = X_raw(2:end, 2:end);
y = X_raw(:, end-2);
protocols = X_raw(:, end-1);
loop = X_raw(:, end);
n_indices = size(X_raw,2)-3;

% Check which variable is used as x in f(x)
ccc = char(data_type);
if ccc(end) == "V"
    X_label = linspace(3, 4.4, n_indices);
    xlabel_name = "V";
elseif ccc(end) == "t"
    X_label = flip(1:n_indices)/n_indices;
    xlabel_name = "\tilde{t}";
end

% Print dummy errors
% cal_MAPE(y(loop == 0), mean(y(loop ~= 0)))
% cal_MAPE(y(loop == 1), mean(y(loop ~= 1)))
% cal_MAPE(y(loop == 2), mean(y(loop ~= 2)))
% cal_MAPE(y(loop == 3), mean(y(loop ~= 3)))
% cal_MAPE(y(loop == 4), mean(y(loop ~= 4)))

%% Analyze the given dataset
% Four dataset used in the article can be found in github
if ~isfile(filename)
    X_raw = xlsread("data_formation/" + data_type + "_all.xlsx");
    X_raw = X_raw(2:end, 2:end);

    if if_logtransform
        beta_name = "log_" + data_type + "_betamatrix_outer";
        lossmatrix_name = "log_" + data_type + "_cv_lossmatrix_outer";
    else
        beta_name = data_type + "_betamatrix_outer";
        lossmatrix_name = data_type + "_cv_lossmatrix_outer";
    end

    % To get the number of lambdas used in the fused lasso work
    lossmatrix_raw = xlsread("results_fusedlasso/" + lossmatrix_name + "0.csv");

    MAPE_train_list = nan * ones(nfolds_outer, nfolds_inner, size(lossmatrix_raw, 2)-1);
    RMSE_train_list = nan * ones(nfolds_outer, nfolds_inner, size(lossmatrix_raw, 2)-1);
    MAPE_test_list = nan * ones(nfolds_outer, nfolds_inner, size(lossmatrix_raw, 2)-1);
    RMSE_test_list = nan * ones(nfolds_outer, nfolds_inner, size(lossmatrix_raw, 2)-1);
    MAPE_dummy = nan * ones(nfolds_outer, nfolds_inner);
    RMSE_dummy = nan * ones(nfolds_outer, nfolds_inner);
    dtw_ratio_beta_list = nan * ones(nfolds_outer, nfolds_inner, size(lossmatrix_raw, 2)-1);
    dtw_ratio_others_list = nan * ones(nfolds_outer, nfolds_inner, size(lossmatrix_raw, 2)-1);
    path_length_list = nan * ones(nfolds_outer, nfolds_inner, size(lossmatrix_raw, 2)-1);
    
    lossmatrix_list = nan * ones(nfolds_outer, nfolds_inner, size(lossmatrix_raw, 2)-1);
    lambda_list = nan * ones(nfolds_outer, size(lossmatrix_raw, 2)-1);
    beta_FL_list = nan * ones(nfolds_outer, nfolds_inner, n_indices, size(lossmatrix_raw, 2)-1);


    for id_outer = 1:nfolds_outer
        lossmatrix_raw = xlsread("results_fusedlasso/" + lossmatrix_name + num2str(id_outer-1) + ".csv");
        n_lambda = sum(max(lossmatrix_raw(2:end, 2:end)) < 1e6);
        for id_inner = 1:nfolds_inner
            beta_raw = xlsread("results_fusedlasso/" + beta_name + num2str(id_outer-1) + "_inner" + num2str(id_inner-1) + ".csv");
            beta_FL_list(id_outer, id_inner, :, 1:n_lambda) = beta_raw(2:end, 2:n_lambda+1);
        end
        lambda_list(id_outer, 1:n_lambda) = lossmatrix_raw(1, 2:n_lambda+1);
        lossmatrix_list(id_outer, :, 1:n_lambda) = lossmatrix_raw(2:end, 2:n_lambda+1);
    end

    for id_outer = 1:nfolds_outer
        data = squeeze(lossmatrix_list(id_outer, :, :));
        lossmatrix = reshape(data(~isnan(data)), nfolds_inner, []);
        n_lambda = size(lossmatrix,2);

        test_id_outer = X_raw(:, end) == id_outer-1;
        train_id_outer = X_raw(:, end) ~= id_outer-1;

        [~, ~, foldid_outer] = unique(X_raw(train_id_outer,end-1));
        foldid_outer = mod(foldid_outer, nfolds_outer);

        for id_inner = 1:nfolds_inner
           
            X = flip(X_raw(:, 1:n_indices), 2);
            y = X_raw(:, end-2);
            protocols = X_raw(:, end-1);

            X_test_outer = X(test_id_outer, :);
            X_train_outer = X(train_id_outer, :);

            test_id = foldid_outer == id_inner-1;
            train_id = foldid_outer ~= id_inner-1;

            X_train = X_train_outer(train_id, :);
            X_test = X_train_outer(test_id, :);

            rescale_factor = max(std(X_train));
            X = X / rescale_factor;

            X_test_outer = X(test_id_outer, :);
            X_train_outer = X(train_id_outer, :);
            X_train = X_train_outer(train_id, :);
            X_test = X_train_outer(test_id, :);

            y = y(train_id_outer);
            y_train = y(train_id);
            y_test = y(test_id);
            y_train_log = log(y(train_id));
            y_test_log = log(y(test_id));
            [~, X_train_C, X_train_S] = normalize(X_train);
            [~, y_train_C, y_train_S] = normalize(y_train);
            [~, y_train_log_C, y_train_log_S] = normalize(y_train_log);
            X_train_cen = X_train - X_train_C;
            X_test_cen = X_test - X_train_C;
            X_train_std = (X_train - X_train_C) ./ X_train_S;
            X_test_std = (X_test - X_train_C) ./ X_train_S;
            y_train_std = (y_train - y_train_C) ./ y_train_S;
            y_test_std = (y_test - y_train_C) ./ y_train_S;
            y_train_log_std = (y_train_log - y_train_log_C) ./ y_train_log_S;
            y_test_log_std = (y_test_log - y_train_log_C) ./ y_train_log_S;

            if if_logtransform
                y_pred_train_cen = exp(X_train_cen * squeeze(beta_FL_list(id_outer, id_inner, :, 1:n_lambda)) * y_train_log_S + y_train_log_C);
                y_pred_test_cen = exp(X_test_cen * squeeze(beta_FL_list(id_outer, id_inner, :, 1:n_lambda)) * y_train_log_S + y_train_log_C);
            else
                y_pred_train_cen = X_train_cen * squeeze(beta_FL_list(id_outer, id_inner, :, 1:n_lambda)) * y_train_S + y_train_C;
                y_pred_test_cen = X_test_cen * squeeze(beta_FL_list(id_outer, id_inner, :, 1:n_lambda)) * y_train_S + y_train_C;
            end

            MAPE_train_list(id_outer, id_inner, 1:n_lambda) = cal_MAPE(y_train, y_pred_train_cen);
            MAPE_test_list(id_outer, id_inner, 1:n_lambda) = cal_MAPE(y_test, y_pred_test_cen);

            RMSE_train_list(id_outer, id_inner, 1:n_lambda) = cal_RMSE(y_train, y_pred_train_cen);
            RMSE_test_list(id_outer, id_inner, 1:n_lambda) = cal_RMSE(y_test, y_pred_test_cen);

            MAPE_dummy(id_outer, id_inner) = cal_MAPE(y_test, mean(y_train));
            RMSE_dummy(id_outer, id_inner) = cal_RMSE(y_test, mean(y_train));

        end

        [dtw_ratio_beta_list(id_outer, :, 1:n_lambda), dtw_ratio_others_list(id_outer, :, 1:n_lambda)] = cal_dtw_ratio(squeeze(beta_FL_list(id_outer, :, :, 1:n_lambda)), nfolds_inner);
        path_length_list(id_outer, :, 1:n_lambda) = cal_path_length(squeeze(beta_FL_list(id_outer, :, :, 1:n_lambda)), nfolds_inner);
    
    end

    save(filename, 'RMSE_test_list', 'MAPE_test_list', 'dtw_ratio_others_list', 'dtw_ratio_beta_list', 'path_length_list', 'lossmatrix_list', 'lambda_list', 'beta_FL_list', 'RMSE_dummy', 'MAPE_dummy')
end

%% Determine whether the blue region exists
load(filename)

chosen_lambda_inds = nan * ones(nfolds_outer, 1);

% For each fold in outer loop
for m = 1:nfolds_outer
    id_outer = m;
    data = squeeze(lossmatrix_list(id_outer, :, :));
    lossmatrix = reshape(data(~isnan(data)), nfolds_inner, []);
    n_lambda = size(lossmatrix,2);
    ind_lambda_init = find(lambda_list(id_outer, :) < max_lambda, 1);
    min_lambda = 10^floor(log10(min(lambda_list(id_outer,:))));

    % Calculate predictiveness metric
    mean_MAPE = mean(squeeze(MAPE_test_list(id_outer, :, 1:n_lambda)));
    [min_MAPE, ind_min_MAPE] = min(mean_MAPE);
    % Predictiveness threshold as in Table S2
    threshold_MAPE = min_MAPE + std(squeeze(MAPE_test_list(id_outer, :, ind_min_MAPE))) / sqrt(nfolds_inner);

    % Calculate robustness and interpretability metrics
    max_dtw_ratio_others = max(squeeze(dtw_ratio_others_list(id_outer, :, 1:n_lambda)));
    mean_path_length = mean(squeeze(path_length_list(id_outer, :, 1:n_lambda)));

    % Check when the alarm is activated w.r.t. different lambda values
    alarm_MAPE = mean_MAPE >= threshold_MAPE;
    alarm_dtw_ratio_others = max_dtw_ratio_others >= threshold_dtw_ratio_others;
    alarm_path_length = mean_path_length >= threshold_path_length;
    alarm_total = (alarm_MAPE + alarm_dtw_ratio_others + alarm_path_length) > 0;

    % Plot blue and red regions as in Figure S5a-c
    figure(); clf();
    t = tiledlayout(3,6);
    nexttile(1,[1 2])
    for id_inner = 1:nfolds_inner
        legend_name = "Inner " + num2str(id_inner);
        plot(lambda_list(id_outer, ind_lambda_init:n_lambda), squeeze(MAPE_test_list(id_outer,id_inner,ind_lambda_init:n_lambda)), 'Color', cmap(id_inner, :), 'LineWidth', 2, 'DisplayName', legend_name);
        hold on;
    end
    set(gca, 'XScale', 'log')
    xlim([min_lambda, max_lambda])
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',ticklabelsize)
    set(gca,'XTickLabelMode','auto')
    yline(threshold_MAPE, '--', 'LineWidth', 4, 'HandleVisibility','off')
    xlabel('\lambda','FontSize', fontsize)
    ylabel('MAPE','FontSize', fontsize)
    grid minor
    plot_dottedline(alarm_MAPE, squeeze(lambda_list(id_outer, :)), xlim, ylim)
    plot_redbox(alarm_MAPE, squeeze(lambda_list(id_outer, :)), xlim, ylim)
    plot_bluebox(alarm_total, squeeze(lambda_list(id_outer, :)), xlim, ylim)

    nexttile(3,[1 2])
    for id_inner = 1:nfolds_inner
        legend_name = "Inner " + num2str(id_inner);
        plot(lambda_list(id_outer, ind_lambda_init:n_lambda), squeeze(dtw_ratio_others_list(id_outer,id_inner,ind_lambda_init:n_lambda)), 'Color', cmap(id_inner, :), 'LineWidth', 2, 'DisplayName', legend_name);
        hold on;
    end
    set(gca, 'XScale', 'log')
    xlim([min_lambda, max_lambda])
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',ticklabelsize)
    set(gca,'XTickLabelMode','auto')
    yline(threshold_dtw_ratio_others, '--', 'LineWidth', 4, 'HandleVisibility','off')
    xlabel('\lambda','FontSize', fontsize)
    ylabel('DTW ratio','FontSize', fontsize)
    grid minor
    plot_dottedline(alarm_dtw_ratio_others, squeeze(lambda_list(id_outer, :)), xlim, ylim)
    plot_redbox(alarm_dtw_ratio_others, squeeze(lambda_list(id_outer, :)), xlim, ylim)
    plot_bluebox(alarm_total, squeeze(lambda_list(id_outer, :)), xlim, ylim)

    nexttile(5,[1 2])
    for id_inner = 1:nfolds_inner
        legend_name = "Inner " + num2str(id_inner);
        plot(lambda_list(id_outer, ind_lambda_init:n_lambda), squeeze(path_length_list(id_outer,id_inner,ind_lambda_init:n_lambda)), 'Color', cmap(id_inner, :), 'LineWidth', 2, 'DisplayName', legend_name);
        hold on;
    end
    set(gca, 'XScale', 'log')
    set(gca, 'YScale', 'log')
    xlim([min_lambda, max_lambda])
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',ticklabelsize)
    set(gca,'XTickLabelMode','auto')
    yline(threshold_path_length, '--', 'LineWidth', 4, 'HandleVisibility','off')
    xlabel('\lambda','FontSize', fontsize)
    ylabel('Path length','FontSize', fontsize) 
    grid minor
    plot_dottedline(alarm_path_length, squeeze(lambda_list(id_outer, :)), xlim, ylim)
    plot_redbox(alarm_path_length, squeeze(lambda_list(id_outer, :)), xlim, ylim)
    plot_bluebox(alarm_total, squeeze(lambda_list(id_outer, :)), xlim, ylim)

    ind_normal = find(alarm_total == 0); 
    if numel(ind_normal) > 0
        chosen_lambda_inds(m) = max(ind_normal);
        lambda_list(id_outer, chosen_lambda_inds(m))
    end

    set(gcf, 'WindowState', 'maximized');

    % If blue region exists
    if numel(ind_normal) > 0

        X_raw = xlsread("data_formation/" + data_type + "_all.xlsx");
        X_raw = X_raw(2:end, 2:end);

        test_id_outer = X_raw(:, end) == id_outer-1;
        train_id_outer = X_raw(:, end) ~= id_outer-1;
        
        X = flip(X_raw(:, 1:n_indices), 2);
        
        protocols = X_raw(:, end-1);
        X_test_outer = X(test_id_outer, :);
        X_train_outer = X(train_id_outer, :);
        y_train_outer = X_raw(train_id_outer, end-2);
        y_test_outer = X_raw(test_id_outer, end-2);
        if if_logtransform
            y_train_outer = log(y_train_outer);
            y_test_outer = log(y_test_outer);
        end

        rescale_factor = max(std(X_train_outer));

        X_train = X_train_outer / rescale_factor;
        X_test = X_test_outer / rescale_factor;
        y_train = y_train_outer;
        y_test = y_test_outer;
        [~, X_train_C, X_train_S] = normalize(X_train);
        [~, y_train_C, y_train_S] = normalize(y_train);
        X_train_cen = X_train - X_train_C;
        X_test_cen = X_test - X_train_C;
        X_train_std = (X_train - X_train_C) ./ X_train_S;
        X_test_std = (X_test - X_train_C) ./ X_train_S;
        y_train_std = (y_train - y_train_C) ./ y_train_S;
        y_test_std = (y_test - y_train_C) ./ y_train_S;

        [~, ~, foldid_outer] = unique(X_raw(train_id_outer,end-1));
        foldid_outer = mod(foldid_outer, nfolds_outer);

        nexttile(7,[2 3])
        ind_blue = find(ind_normal);
        lambda_show = ind_normal(round(linspace(1, length(ind_normal), num_lambda_show)));
        lambda_ind = chosen_lambda_inds(m);
        cmap_black = gray(length(lambda_show)+1);
        
        % Plot beta's as a function of lambda as in Figure S5d
        for kk = 1:length(lambda_show)
            lambda = lambda_list(id_outer, lambda_show(kk));
            cvx_begin
                variable beta_FL(n_indices,1)
                minimize(0.5*sum((y_train_std - X_train_cen * beta_FL).^2) + lambda * norm(beta_FL(1:n_indices-1) - beta_FL(2:n_indices), 1))
            cvx_end
            if kk == length(lambda_show)
                plot(X_label, beta_FL, 'Color', cmap_black(length(lambda_show)+1-kk, :), 'LineWidth', 5, 'DisplayName', "\beta at \lambda = " + num2str(round(lambda, 4))) 
            else
                plot(X_label, beta_FL, 'Color', cmap_black(length(lambda_show)+1-kk, :), 'LineWidth', 2, 'DisplayName', "\beta at \lambda = " + num2str(round(lambda, 4))) 
            end
            hold on
        end

        yline(0, '--', 'HandleVisibility','off')
        xlim([min(X_label), max(X_label)])
        ylim([plot_ylim(1), plot_ylim(2)])
        grid minor
        lgd = legend('show', 'location', legend_position);
        lgd.NumColumns = 2;
        set(lgd, 'FontSize',13)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',ticklabelsize)
        set(gca,'XTickLabelMode','auto')
        if ccc(end) == "V"
            xlabel("Voltage (V)", 'FontSize', fontsize);
        else
            xlabel("$\tilde{t}$", 'Interpreter', 'latex', 'FontSize', fontsize);
        end
        ylabel('\beta', 'FontSize', fontsize);

        % Plot beta's at each inner loop with the optimal lambda as in Figure S5e
        nexttile(10,[2 3])
        for id_inner = 1:nfolds_inner
            legend_name = "\beta^{(" + num2str(id_inner) + ")} at \lambda = " + num2str(round(lambda_list(id_outer, max(ind_normal)), 4));
            plot(X_label, squeeze(beta_FL_list(id_outer, id_inner, :, max(ind_normal))), 'Color', cmap(id_inner, :), 'LineWidth', 2, 'DisplayName', legend_name);
            hold on
        end
        yline(0, '--', 'HandleVisibility','off')
        xlim([min(X_label), max(X_label)])
        ylim([plot_ylim(1), plot_ylim(2)])
        grid minor
        lgd = legend('show', 'location', legend_position);
        set(lgd, 'FontSize',13)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',ticklabelsize)
        set(gca,'XTickLabelMode','auto')
        if ccc(end) == "V"
            xlabel("Voltage (V)", 'FontSize', fontsize);
        else
            xlabel("$\tilde{t}$", 'Interpreter', 'latex', 'FontSize', fontsize);
        end
        ylabel('\beta', 'FontSize', fontsize);
    else
        % Tell there's no lambda in blue region
        nexttile(7,[2 3])
        box on
        text(0.5, 0.5, 'No \lambda in the blue region', 'FontSize', 20', 'FontWeight', 'Bold', 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'middle' )
        nexttile(10,[2 3])
        box on
        text(0.5, 0.5, 'No \lambda in the blue region', 'FontSize', 20', 'FontWeight', 'Bold', 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'middle' )
    end

end

MAPE_train_final = nan * ones(nfolds_outer,1);
MAPE_test_final = nan * ones(nfolds_outer,1);
RMSE_train_final = nan * ones(nfolds_outer,1);
RMSE_test_final = nan * ones(nfolds_outer,1);

%% Proceed feature design for the outer loops where the blue region exists
for m = 1:nfolds_outer
    id_outer = m;
    chosen_lambda_ind = chosen_lambda_inds(m);
    
    % If the blue region exists
    if ~isnan(chosen_lambda_ind)

        X_raw = xlsread("data_formation/" + data_type + "_all.xlsx");
        X_raw = X_raw(2:end, 2:end);

        test_id_outer = X_raw(:, end) == id_outer-1;
        train_id_outer = X_raw(:, end) ~= id_outer-1;
        X = flip(X_raw(:, 1:n_indices), 2);
        
        protocols = X_raw(:, end-1);
        X_test_outer = X(test_id_outer, :);
        X_train_outer = X(train_id_outer, :);
        y_train_outer = X_raw(train_id_outer, end-2);
        y_test_outer = X_raw(test_id_outer, end-2);
        if if_logtransform
            y_train_outer = log(y_train_outer);
            y_test_outer = log(y_test_outer);
        end

        rescale_factor = max(std(X_train_outer));

        X_train = X_train_outer / rescale_factor;
        X_test = X_test_outer / rescale_factor;
        y_train = y_train_outer;
        y_test = y_test_outer;
        [~, X_train_C, X_train_S] = normalize(X_train);
        [~, y_train_C, y_train_S] = normalize(y_train);
        X_train_cen = X_train - X_train_C;
        X_test_cen = X_test - X_train_C;
        X_train_std = (X_train - X_train_C) ./ X_train_S;
        X_test_std = (X_test - X_train_C) ./ X_train_S;
        y_train_std = (y_train - y_train_C) ./ y_train_S;
        y_test_std = (y_test - y_train_C) ./ y_train_S;

        [~, ~, foldid_outer] = unique(X_raw(train_id_outer,end-1));
        foldid_outer = mod(foldid_outer, nfolds_outer);
        
        % Reconstruct beta using all training set of the outer fold at the determined lambda
        lambda_ind = chosen_lambda_ind;
        lambda = lambda_list(id_outer, lambda_ind);
        cvx_begin
            variable beta_FL(n_indices,1)
            minimize(0.5*sum((y_train_std - X_train_cen * beta_FL).^2) + lambda * norm(beta_FL(1:n_indices-1) - beta_FL(2:n_indices), 1))
        cvx_end
        
        % Find partition boundaries as in Figure 3A
        x0_filtered = find_x0_filtered(beta_FL);
        x0 = [1; x0_filtered; n_indices];
        round(X_label(x0), 2)
        
        % Provide labels for each boundary
        alphabetlist = string(mat2cell('A':'Z',1,ones(1,26)));
        xticklabelname = alphabetlist;
        for j = 1:26
            xticklabelname = [xticklabelname, alphabetlist(j) + alphabetlist];
        end
        xticklabelname = [{"3.0V"}, xticklabelname{1:(length(x0)-2)}, {"4.4V"}];
        
        % Merge sections if the boundary is not necessary. Iterate until there is no more boundary to remove.
        [xticklabelname_new, x0_new] = determine_merge(X_train_outer, x0, foldid_outer, X_raw, beta_FL, xticklabelname, merge_threshold, true, nfolds_inner);

        while length(x0_new) < length(x0)
            x0 = x0_new;
            xticklabelname = xticklabelname_new;
            [xticklabelname_new, x0_new] = determine_merge(X_train_outer, x0, foldid_outer, X_raw, beta_FL, xticklabelname, merge_threshold, false, nfolds_inner);
        end
        [xticklabelname_merge, x0_merge] = determine_merge(X_train_outer, x0, foldid_outer, X_raw, beta_FL, xticklabelname, merge_threshold, true, nfolds_inner);
        round(X_label(x0_merge), 2)
       
        % Generate "difference" and "mean" features for the final boundaries
        X_features_train = [X_train_outer(:,x0_merge(1)) - X_train_outer(:,x0_merge(2)), mean(X_train_outer(:,x0_merge(1):x0_merge(2)),2)];
        X_features_test = [X_test_outer(:,x0_merge(1)) - X_test_outer(:,x0_merge(2)), mean(X_test_outer(:,x0_merge(1):x0_merge(2)),2)];
        featurenames = {"diff_" + xticklabelname_merge{1} + "_" + xticklabelname_merge{2}, "mean_" + xticklabelname_merge{1} + "_" + xticklabelname_merge{2}};
        featurenames_V = {"diff_3.0V_" + num2str(round(X_label(x0(2)), 2)) + "V", "mean_3.0V_" + num2str(round(X_label(x0(2)), 2)) + "V"};
        for i = 2:length(x0_merge)-1
            X_features_train = [X_features_train, X_train_outer(:,x0_merge(i)) - X_train_outer(:,x0_merge(i+1)), mean(X_train_outer(:,x0_merge(i):x0_merge(i+1)),2)];
            X_features_test = [X_features_test, X_test_outer(:,x0_merge(i)) - X_test_outer(:,x0_merge(i+1)), mean(X_test_outer(:,x0_merge(i):x0_merge(i+1)),2)];
            featurenames = [featurenames, {"diff_" + xticklabelname_merge{i} + "_" + xticklabelname_merge{i+1}}, {"mean_" + xticklabelname_merge{i} + "_" + xticklabelname_merge{i+1}}];
            featurenames_V = [featurenames_V, {"diff_" + num2str(round(X_label(x0(i)), 2)) + "V_" + num2str(round(X_label(x0(i+1)), 2)) + "V"}, {"mean_" + num2str(round(X_label(x0(i)), 2)) + "V_" + num2str(round(X_label(x0(i+1)), 2)) + "V"}];
        end

        R = corrcoef([X_features_train, y_train_outer]);
        [X_features_train_final, X_features_test_final, featurenames_final, featurenames_V_final] = downselect_features(R, X_features_train, X_features_test, featurenames, featurenames_V, corr_threshold);
        featurenames_V_final
        
        % Save the obtained feature matrix
        if if_savematrix
            if ~exist("Features_designed", 'dir')
                mkdir("Features_designed");
            end
            if if_logtransform
                writematrix([X_features_train_final, y_train_outer], "Features_designed/log_" + data_type + "_train_Outer_" + num2str(id_outer) + ".xlsx")
                writematrix([X_features_test_final, y_test_outer], "Features_designed/log_" + data_type + "_test_Outer_" + num2str(id_outer) + ".xlsx")
                writematrix(protocols(train_id_outer), "Features_designed/log_" + data_type + "_grouplabel_Outer_" + num2str(id_outer) + ".xlsx")
            else
                writematrix([X_features_train_final, y_train_outer], "Features_designed/" + data_type + "_train_Outer_" + num2str(id_outer) + ".xlsx")
                writematrix([X_features_test_final, y_test_outer], "Features_designed/" + data_type + "_test_Outer_" + num2str(id_outer) + ".xlsx")
                writematrix(protocols(train_id_outer), "Features_designed/" + data_type + "_grouplabel_Outer_" + num2str(id_outer) + ".xlsx")
            end
        end
        
        X = X_raw(train_id_outer, 1:n_indices);
        dXdV = [zeros(size(X,1), 1), X(:, 1:end-1) - X(:, 2:end)];
        d2XdV2 = [zeros(size(dXdV,1), 1), dXdV(:, 1:end-1) - dXdV(:, 2:end)];
        
        if ccc(end) == "V"
            X = flip(X,2);
            dXdV = flip(dXdV,2);
            d2XdV2 = flip(d2XdV2,2);
        end
        
        X_smooth = movmean(X, 10, 2);
        dXdV_smooth = movmean(dXdV, 10, 2);
        d2XdV2_smooth = movmean(d2XdV2, 10, 2);
       
        N = size(X,1);
        cmap_cl = jet(N);
        color_idx = zeros(N,1);
        for k = 1:N
            color_idx(k) = round((y_train_outer(k) - min(y_train_outer)) / (max(y_train_outer) - min(y_train_outer)) * (N - 1)) + 1;
        end
        x0_diff = setdiff(x0_filtered, x0_merge);
        
        % Plot the location of final boundaries on top of f(x), df/dx, and d^2f/dx^2 curves as in Figure 3B-D
        figure(); clf();
        subplot(3,1,1);
        box on
        hold on
        for k = 1:size(dXdV,1)
            plot(X_label, X(k, :), 'Color', cmap_cl(color_idx(k),:))
        end
        if id_outer == 1 && data_type == "B_Q_V"
            for dd = 1:length(x0_merge)
                xline(X_label(x0_merge(dd)), '--', 'LineWidth', 0.5, 'HandleVisibility','off')
            end
            xline(X_label(x0_merge(11)), '--', 'LineWidth', 3, 'HandleVisibility','off')
            xline(X_label(x0_merge(12)), '--', 'LineWidth', 3, 'HandleVisibility','off')
            xline(X_label(x0_merge(13)), '--', 'LineWidth', 3, 'HandleVisibility','off')
        end
        xlim([min(X_label), max(X_label)])
        set(gca,'XTickLabelMode','auto')
        xlabel('Voltage (V)')
        ylabel("Q^B (Ah)")
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',ticklabelsize)
        
        subplot(3,1,2);
        box on
        hold on
        for k = 1:size(dXdV,1)
            plot(X_label, dXdV_smooth(k, :), 'Color', cmap_cl(color_idx(k),:))
        end
        if id_outer == 1 && data_type == "B_Q_V"
            for dd = 1:length(x0_merge)
                xline(X_label(x0_merge(dd)), '--', 'LineWidth', 0.5, 'HandleVisibility','off')
            end
            xline(X_label(x0_merge(11)), '--', 'LineWidth', 3, 'HandleVisibility','off')
            xline(X_label(x0_merge(12)), '--', 'LineWidth', 3, 'HandleVisibility','off')
            xline(X_label(x0_merge(13)), '--', 'LineWidth', 3, 'HandleVisibility','off')
        end
        
        xlim([min(X_label), max(X_label)])
        set(gca,'XTickLabelMode','auto')
        xlabel('Voltage (V)')
        ylabel("dQ^B/dV (Ah/V)")
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',ticklabelsize)
        xlim([min(X_label), max(X_label)])
        
        subplot(3,1,3);
        box on
        hold on
        for k = 1:size(dXdV,1)
            plot(X_label, d2XdV2_smooth(k, :), 'Color', cmap_cl(color_idx(k),:))
        end
        if id_outer == 1 && data_type == "B_Q_V"
            for dd = 1:length(x0_merge)
                xline(X_label(x0_merge(dd)), '--', 'LineWidth', 0.5, 'HandleVisibility','off')
            end
            xline(X_label(x0_merge(11)), '--', 'LineWidth', 3, 'HandleVisibility','off')
            xline(X_label(x0_merge(12)), '--', 'LineWidth', 3, 'HandleVisibility','off')
            xline(X_label(x0_merge(13)), '--', 'LineWidth', 3, 'HandleVisibility','off')
        end
        
        yline(0, '--', 'HandleVisibility','off')
        plot(X_label, mean(d2XdV2_smooth), 'k', 'LineWidth', 3)
        xlim([min(X_label), max(X_label)])
        set(gca,'XTickLabelMode','auto')
        ylim([-2e-5, 2e-5])
        xlabel('Voltage (V)')
        ylabel("d^2Q^B/dV^2 (Ah/V^2)")
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',ticklabelsize)

        set(gcf, 'WindowState', 'maximized');

        % Plot Figure 4
        axes1 = axes('Parent',figure);
        box on
        yyaxis left
        hold on
        for k = 1:size(X,1)
            plot(X_label, (X(k, :) - mean(X, 1)) / max(std(X)), '-', 'Color', cmap_cl(color_idx(k),:))
        end
        if id_outer == 1
            for i = 1:length(x0_filtered)
                xline(X_label(x0_filtered(i)), '--', 'LineWidth', 2, 'HandleVisibility','off')
            end
        else
            for i = 1:length(x0_merge)
                xline(X_label(x0_merge(i)), '--', 'LineWidth', 2, 'HandleVisibility','off')
            end
        end
        
        xlabel('Voltage (V)', 'FontSize', 30)
        ylim([-3, 3])
        ylabel('Standardized Q^B(V)', 'FontSize', 30)
        yyaxis right
        plot(X_label, beta_FL, 'k', 'LineWidth', 4)
        set(axes1,'YColor',[0 0 0]);
        ylabel('\beta', 'FontSize', 30)
        ylim([plot_ylim(1), plot_ylim(2)])
        yline(0, '--', 'LineWidth', 2, 'HandleVisibility','off')
        xlim([min(X_label), max(X_label)])
        set(gca,'XTickLabelMode','auto')
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',30)
        set(gcf, 'WindowState', 'maximized');        
        
        colormap(jet);

        [X_train_final, X_train_final_C, X_train_final_S] = normalize(X_features_train_final);
        [y_train_final, y_train_final_C, y_train_final_S] = normalize(y_train_outer);
  
        X_test_final = (X_features_test_final - X_train_final_C) ./ X_train_final_S;
        y_test_final = (y_test_outer - y_train_final_C) ./ y_train_final_S;

        mdl_final = fitlm(X_train_final,y_train_final);
    
        y_pred_train_final = exp(predict(mdl_final, X_train_final) * y_train_final_S + y_train_final_C);
        y_pred_test_final = exp(predict(mdl_final, X_test_final) * y_train_final_S + y_train_final_C);
        
        MAPE_train_final(m, 1) = cal_MAPE(exp(y_train_outer), y_pred_train_final);
        MAPE_test_final(m, 1) = cal_MAPE(exp(y_test_outer), y_pred_test_final);
        RMSE_train_final(m, 1) = cal_RMSE(exp(y_train_outer), y_pred_train_final);
        RMSE_test_final(m, 1) = cal_RMSE(exp(y_test_outer), y_pred_test_final);

    end
end

%% Functions
function MAPE = cal_MAPE(y, y_pred)
    % Calculate MAPE when y and y_pred are given
    MAPE = mean(abs(y - y_pred) ./ y) * 100;
end

function RMSE = cal_RMSE(y, y_pred)
    % Calculate RMSE when y and y_pred are given
    RMSE = sqrt(mean((y - y_pred).^2));
end


function [dtw_ratio_beta, dtw_ratio_others] = cal_dtw_ratio(beta_list, nfolds_inner)
    % Calculate the ratio of dynamic time warping distance (i.e., robustness metric)
    n_lambda = size(beta_list,3);
    dtw_list = nan * ones(nfolds_inner,n_lambda);
    dtw_0_list = nan * ones(nfolds_inner,n_lambda);
    dtw_0_others_list = nan * ones(nfolds_inner,n_lambda);
    for id_lambda = 1:n_lambda
        for id_inner = 1:nfolds_inner
            dtw_list(id_inner, id_lambda) = dtw(beta_list(id_inner, :, id_lambda),mean(beta_list(1:nfolds_inner ~= id_inner, :, id_lambda), 1));
            dtw_0_list(id_inner, id_lambda) = dtw(beta_list(id_inner, :, id_lambda),zeros(size(beta_list,2), 1));
            dtw_0_others_list(id_inner, id_lambda) = dtw(mean(beta_list(1:nfolds_inner ~= id_inner, :, id_lambda), 1), zeros(size(beta_list,2), 1));
        end
    end
    dtw_ratio_beta = dtw_list ./ dtw_0_list;
    dtw_ratio_others = dtw_list ./ dtw_0_others_list;
    dtw_ratio_beta = movmean(dtw_ratio_beta, 5);
    dtw_ratio_others = movmean(dtw_ratio_others, 5);
end

function x0_filtered = find_x0_filtered(beta)
    % Find where the jump occurred in beta
    im_x0 = abs(diff(beta)) >= 0.001 * (max(beta) - min(beta));
    x0_filtered = find(im_x0);
end

function path_length = cal_path_length(beta_list, nfolds_inner)
    % Calculate path length (i.e., interpretability metric)
    n_lambda = size(beta_list,3);
    path_length = nan * ones(nfolds_inner, n_lambda);
    for id_lambda = 1:n_lambda
        for id_inner = 1:nfolds_inner
            path_length(id_inner, id_lambda) = sum(abs(diff(beta_list(id_inner, :, id_lambda))));
        end
    end
end

function plot_redbox(alarm, lambdas, xl, yl)
    % Plot red box as in Figures S5a-c
    CC = bwconncomp(alarm);
    for k = 1:CC.NumObjects
        if length(CC.PixelIdxList{1, k}) > 1
            if min(CC.PixelIdxList{1, k}) == 1
                if max(CC.PixelIdxList{1, k}) == length(alarm)
                    r = rectangle('Position', [xl(1),yl(1),xl(2) - xl(1),yl(2)-yl(1)],'FaceColor', [1, 0, 0, 0.2],'EdgeColor', [1, 0, 0, 0.2]);
                else
                    xmin = 0.5*(lambdas(max(CC.PixelIdxList{1, k})) + lambdas(max(CC.PixelIdxList{1, k})+1));
                    r = rectangle('Position', [xmin,yl(1),max(xl(2) - xmin, 1),yl(2)-yl(1)],'FaceColor', [1, 0, 0, 0.2],'EdgeColor', [1, 0, 0, 0.2]);
                end
            else
                if max(CC.PixelIdxList{1, k}) == length(alarm)
                    xmax = 0.5*(lambdas(min(CC.PixelIdxList{1, k})) + lambdas(min(CC.PixelIdxList{1, k})-1));
                    r = rectangle('Position', [xl(1),yl(1),xmax - xl(1),yl(2)-yl(1)],'FaceColor', [1, 0, 0, 0.2],'EdgeColor', [1, 0, 0, 0.2]);
                else
                    xmin = 0.5*(lambdas(max(CC.PixelIdxList{1, k})) + lambdas(max(CC.PixelIdxList{1, k})+1));
                    xmax = 0.5*(lambdas(min(CC.PixelIdxList{1, k})) + lambdas(min(CC.PixelIdxList{1, k})-1));
                    r = rectangle('Position', [xmin,yl(1),xmax - xmin,yl(2)-yl(1)],'FaceColor', [1, 0, 0, 0.2],'EdgeColor', [1, 0, 0, 0.2]);
                end
            end
        end
    end
    xlim([xl(1), xl(2)]);
    ylim([yl(1), yl(2)]);
end

function plot_bluebox(alarm, lambdas, xl, yl)
    % Plot blue box as in Figures S5a-c
    CC = bwconncomp(~alarm);
    for k = 1:CC.NumObjects
        if length(CC.PixelIdxList{1, k}) > 1
            if min(CC.PixelIdxList{1, k}) == 1
                if max(CC.PixelIdxList{1, k}) == length(alarm)
                    r = rectangle('Position', [xl(1),yl(1),xl(2) - xl(1),yl(2)-yl(1)],'FaceColor', [0, 0, 1, 0.3],'EdgeColor', [0, 0, 1, 0.3]);
                else
                    xmin = 0.5*(lambdas(max(CC.PixelIdxList{1, k})) + lambdas(max(CC.PixelIdxList{1, k})+1));
                    r = rectangle('Position', [xmin,yl(1),max(xl(2) - xmin, 1),yl(2)-yl(1)],'FaceColor', [0, 0, 1, 0.3],'EdgeColor', [0, 0, 1, 0.3]);
                end
            else
                if max(CC.PixelIdxList{1, k}) == length(alarm)
                    xmax = 0.5*(lambdas(min(CC.PixelIdxList{1, k})) + lambdas(min(CC.PixelIdxList{1, k})-1));
                    r = rectangle('Position', [xl(1),yl(1),xmax - xl(1),yl(2)-yl(1)],'FaceColor', [0, 0, 1, 0.3],'EdgeColor', [0, 0, 1, 0.3]);
                else
                    xmin = 0.5*(lambdas(max(CC.PixelIdxList{1, k})) + lambdas(max(CC.PixelIdxList{1, k})+1));
                    xmax = 0.5*(lambdas(min(CC.PixelIdxList{1, k})) + lambdas(min(CC.PixelIdxList{1, k})-1));
                    r = rectangle('Position', [xmin,yl(1),xmax - xmin,yl(2)-yl(1)],'FaceColor', [0, 0, 1, 0.3],'EdgeColor', [0, 0, 1, 0.3]);
                end
            end
        end
    end
    xlim([xl(1), xl(2)]);
    ylim([yl(1), yl(2)]);
end

function plot_dottedline(alarm, lambdas, xl, yl)
    % Plot the boundaries of red and blue boxes in Figure S5
    ind_lambda = find(diff(alarm > 0));
    for k = 1:length(ind_lambda)
        xline(0.5*(lambdas(ind_lambda) + lambdas(ind_lambda+1)), '--', 'LineWidth', 2, 'HandleVisibility','off')
    end
    xlim([xl(1), xl(2)]);
    ylim([yl(1), yl(2)]);
end


function [xticklabelname, x0] = determine_merge(X_train_outer, x0, foldid_outer, X_raw, beta_avg, xticklabelname, merge_threshold, if_plot, nfolds_inner)
    % Merge sections as in Algorithm S1
    MAPE_train_list_1 = nan * ones(nfolds_inner, length(x0)-1);
    RMSE_train_list_1 = nan * ones(nfolds_inner, length(x0)-1);
    NRMSE_train_list_1 = nan * ones(nfolds_inner, length(x0)-1);
    MAPE_test_list_1 = nan * ones(nfolds_inner, length(x0)-1);
    RMSE_test_list_1 = nan * ones(nfolds_inner, length(x0)-1);
    NRMSE_test_list_1 = nan * ones(nfolds_inner, length(x0)-1);
    
    MAPE_train_list_2 = nan * ones(nfolds_inner, length(x0)-2);
    RMSE_train_list_2 = nan * ones(nfolds_inner, length(x0)-2);
    NRMSE_train_list_2 = nan * ones(nfolds_inner, length(x0)-2);
    MAPE_test_list_2 = nan * ones(nfolds_inner, length(x0)-2);
    RMSE_test_list_2 = nan * ones(nfolds_inner, length(x0)-2);
    NRMSE_test_list_2 = nan * ones(nfolds_inner, length(x0)-2);

    for k = 1:length(x0)-1
        ind_init = x0(k);
        ind_end = x0(k+1);
        
        X_selected = X_train_outer(:, ind_init:ind_end);
        X_features = [X_selected(:,end) - X_selected(:,1), mean(X_selected,2)];        

        for id_inner = 1:nfolds_inner
            
            test_id = foldid_outer == id_inner-1;
            train_id = foldid_outer ~= id_inner-1;

            rescale_factor = max(std(X_train_outer(train_id, :)));
            beta = beta_avg / rescale_factor;
            y = (X_selected - mean(X_selected(train_id,:))) * beta(ind_init:ind_end);
    
            X_train = X_features(train_id, :);
            X_test = X_features(test_id, :);
            y_train = y(train_id);
            y_test = y(test_id);
            y_train_log = log(y(train_id));
            y_test_log = log(y(test_id));
            [~, X_train_C, X_train_S] = normalize(X_train);
            [~, y_train_C, y_train_S] = normalize(y_train);
            [~, y_train_log_C, y_train_log_S] = normalize(y_train_log);
            X_train_cen = X_train - X_train_C;
            X_test_cen = X_test - X_train_C;
            y_train_std = (y_train - y_train_C) ./ y_train_S;
            
            mdl = fitlm(X_train_cen,y_train_std);
        
            y_pred_train_cen = predict(mdl, X_train_cen) * y_train_S + y_train_C;
            y_pred_test_cen = predict(mdl, X_test_cen) * y_train_S + y_train_C;
            
            MAPE_train_list_1(id_inner, k) = cal_MAPE(y_train, y_pred_train_cen);
            MAPE_test_list_1(id_inner, k) = cal_MAPE(y_test, y_pred_test_cen);
        
            RMSE_train_list_1(id_inner, k) = cal_RMSE(y_train, y_pred_train_cen);
            RMSE_test_list_1(id_inner, k) = cal_RMSE(y_test, y_pred_test_cen);
        
            NRMSE_train_list_1(id_inner, k) = RMSE_train_list_1(id_inner, k) / (max(y_train) - min(y_train));
            NRMSE_test_list_1(id_inner, k) = RMSE_test_list_1(id_inner, k) / (max(y_test) - min(y_test));
        end
    end

    for k = 1:length(x0)-2
        ind_init = x0(k);
        ind_end = x0(k+2);
        
        X_selected = X_train_outer(:, ind_init:ind_end);
        X_features = [X_selected(:,end) - X_selected(:,1), mean(X_selected,2)];

        for id_inner = 1:nfolds_inner
            
            test_id = foldid_outer == id_inner-1;
            train_id = foldid_outer ~= id_inner-1;

            rescale_factor = max(std(X_train_outer(train_id, :)));
            beta = beta_avg / rescale_factor;
            y = (X_selected - mean(X_selected(train_id,:))) * beta(ind_init:ind_end);
    
            X_train = X_features(train_id, :);
            X_test = X_features(test_id, :);
            y_train = y(train_id);
            y_test = y(test_id);
            y_train_log = log(y(train_id));
            y_test_log = log(y(test_id));
            [~, X_train_C, X_train_S] = normalize(X_train);
            [~, y_train_C, y_train_S] = normalize(y_train);
            [~, y_train_log_C, y_train_log_S] = normalize(y_train_log);
            X_train_cen = X_train - X_train_C;
            X_test_cen = X_test - X_train_C;
            y_train_std = (y_train - y_train_C) ./ y_train_S;
            
            mdl = fitlm(X_train_cen,y_train_std);
        
            y_pred_train_cen = predict(mdl, X_train_cen) * y_train_S + y_train_C;
            y_pred_test_cen = predict(mdl, X_test_cen) * y_train_S + y_train_C;
            
            MAPE_train_list_2(id_inner, k) = cal_MAPE(y_train, y_pred_train_cen);
            MAPE_test_list_2(id_inner, k) = cal_MAPE(y_test, y_pred_test_cen);
        
            RMSE_train_list_2(id_inner, k) = cal_RMSE(y_train, y_pred_train_cen);
            RMSE_test_list_2(id_inner, k) = cal_RMSE(y_test, y_pred_test_cen);
        
            NRMSE_train_list_2(id_inner, k) = RMSE_train_list_2(id_inner, k) / (max(y_train) - min(y_train));
            NRMSE_test_list_2(id_inner, k) = RMSE_test_list_2(id_inner, k) / (max(y_test) - min(y_test));
        end
    end

    if if_plot
        figure(); clf();
        bar(1:(length(x0)-1),mean(RMSE_test_list_1), 0.5);
        hold on
        bar((1:(length(x0)-2)) + 0.5,mean(RMSE_test_list_2), 0.5);
        yline(merge_threshold, '--', 'LineWidth', 4)
        
        ylabel('RMSE', 'FontSize', 30);
        xlabel('Section', 'FontSize', 30);
        set(gca,'Xtick',0.5:(length(x0)-0.5),'XTickLabel',xticklabelname)
        a = get(gca,'XTickLabel');
        set(gca,'XTickLabel',a,'fontsize',40)
        
        set(gcf, 'WindowState', 'maximized');
    end
    
    [~, ind_merge] = min(mean(RMSE_test_list_2));
    if min(mean(RMSE_test_list_2)) < merge_threshold
        xticklabelname(ind_merge + 1) = [];
        x0(ind_merge + 1) = [];
    end
end

function [X_features_train, X_features_test, featurenames, featurenames_V] = downselect_features(R, X_features_train, X_features_test, featurenames, featurenames_V, corr_threshold)
    % Downselect features as in Algorithm S2
    ind_features = [];
    ind_keep = find(abs(R(:, end)) >= corr_threshold);
    R_reduced = R(ind_keep, ind_keep);
    while size(R_reduced, 1) > 1
        [~, feature_ind] = max(abs(R_reduced(1:end-1, end)));
        ind_keep = find(abs(R_reduced(1:end-1, feature_ind)) <= 0.2);
        ind_keep = [ind_keep; size(R_reduced, 1)];
        ind_features = [ind_features, find(abs(R(:, end)) == max(abs(R_reduced(1:end-1, end))))];
        R_reduced = R_reduced(ind_keep, ind_keep);
    end
    featurenames = featurenames(ind_features);
    featurenames_V = featurenames_V(ind_features);
    X_features_train = X_features_train(:, ind_features);
    X_features_test = X_features_test(:, ind_features);
end
