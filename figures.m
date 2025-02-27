close all; clear; clc;
% Scatter plot of RMSE vs. MAPE of autoML models for promisingness of the input data
[~, ~, X_autoML] = xlsread("summarized_results/autoML_results.csv");
X_autoML = X_autoML(2:end, :);
RMSE_xlim_end = 400;
MAPE_xlim_end = 40;
X_autoML = sortrows(X_autoML, 1);
ind_best_autoML = 1687;

ind_A_Q_V = X_autoML(:,2) == "A" & X_autoML(:,3) == "Q_V";
ind_A_t_V = X_autoML(:,2) == "A" & X_autoML(:,3) == "t_V";
ind_B_Q_V = X_autoML(:,2) == "B" & X_autoML(:,3) == "Q_V";
ind_B_V_t = X_autoML(:,2) == "B" & X_autoML(:,3) == "V_t";
ind_C_Q_V = X_autoML(:,2) == "C" & X_autoML(:,3) == "Q_V";
ind_C_V_t = X_autoML(:,2) == "C" & X_autoML(:,3) == "V_t";

[~, ~, X_agnostic] = xlsread("summarized_results/agnostic_results.csv");
rank_agnostic = cell2mat(X_agnostic(2:end,32)) + cell2mat(X_agnostic(2:end,36));
[~, order_rank_agnostic] = sort(rank_agnostic);
ind_agnostic = order_rank_agnostic(1) + 1;
agnostic_mean_MAPE = cell2mat(X_agnostic(ind_agnostic, 28));
agnostic_med_MAPE = cell2mat(X_agnostic(ind_agnostic, 32));
agnostic_max_MAPE = cell2mat(X_agnostic(ind_agnostic, 36));
agnostic_HL_MAPE = cell2mat(X_agnostic(ind_agnostic, 40));
agnostic_mean_RMSE = cell2mat(X_agnostic(ind_agnostic, 26));
agnostic_med_RMSE = cell2mat(X_agnostic(ind_agnostic, 30));
agnostic_max_RMSE = cell2mat(X_agnostic(ind_agnostic, 34));
agnostic_HL_RMSE = cell2mat(X_agnostic(ind_agnostic, 38));

[~, ~, X_designed] = xlsread("summarized_results/designed_results.csv");
rank_designed = cell2mat(X_designed(2:end,31)) + cell2mat(X_designed(2:end,35));
[~, order_rank_designed] = sort(rank_designed);
ind_designed = order_rank_designed(1) + 1;
designed_mean_MAPE = cell2mat(X_designed(ind_designed, 27));
designed_med_MAPE = cell2mat(X_designed(ind_designed, 31));
designed_max_MAPE = cell2mat(X_designed(ind_designed, 35));
designed_HL_MAPE = cell2mat(X_designed(ind_designed, 39));
designed_mean_RMSE = cell2mat(X_designed(ind_designed, 25));
designed_med_RMSE = cell2mat(X_designed(ind_designed, 29));
designed_max_RMSE = cell2mat(X_designed(ind_designed, 33));
designed_HL_RMSE = cell2mat(X_designed(ind_designed, 37));

figure(); clf();
hold on
yline(agnostic_max_MAPE,'--', 'LineWidth', 3, 'HandleVisibility','off');
xline(agnostic_med_MAPE,'--', 'LineWidth', 3, 'HandleVisibility','off');
plot(linspace(8, 18, 10), linspace(agnostic_med_MAPE + agnostic_max_MAPE - 8, agnostic_med_MAPE + agnostic_max_MAPE - 18, 10), '--r', 'LineWidth', 3, 'HandleVisibility','off')
scatter(cell2mat(X_autoML(ind_A_Q_V, 35)), cell2mat(X_autoML(ind_A_Q_V, 39)), 50, [0 0.4470 0.5410], 'filled')
scatter(cell2mat(X_autoML(ind_A_t_V, 35)), cell2mat(X_autoML(ind_A_t_V, 39)), 50, [0.6500 0.3250 0.0980], 'filled')
scatter(cell2mat(X_autoML(ind_B_Q_V, 35)), cell2mat(X_autoML(ind_B_Q_V, 39)), 70, [0.5 0.5 0.5], 'x', 'LineWidth', 2)
scatter(cell2mat(X_autoML(ind_B_V_t, 35)), cell2mat(X_autoML(ind_B_V_t, 39)), 70, 'g', 'x', 'LineWidth', 2)
scatter(cell2mat(X_autoML(ind_C_Q_V, 35)), cell2mat(X_autoML(ind_C_Q_V, 39)), 50, 'm', '^', 'LineWidth', 1)
scatter(cell2mat(X_autoML(ind_C_V_t, 35)), cell2mat(X_autoML(ind_C_V_t, 39)), 50, 'c', '^', 'LineWidth', 1)
scatter(agnostic_med_MAPE, agnostic_max_MAPE, 800, 'r', 'filled', 'pentagram', 'LineWidth', 1)
scatter(designed_med_MAPE, designed_max_MAPE, 800, 'b', 'filled', 'pentagram', 'LineWidth', 1)
scatter(cell2mat(X_autoML(ind_best_autoML, 35)), cell2mat(X_autoML(ind_best_autoML, 39)), 400, 'k', 'x', 'LineWidth', 4)
xlim([8, 18])
ylim([10, 20])
%legend({'$Q^{\rm{A}}(V)$', '$t^{\rm{A}}(V)$', '$Q^{\rm{B}}(V)$' '$V^{\rm{B}}(\tilde{t})$' '$Q^{\rm{C}}(V)$', '$V^{\rm{C}}(\tilde{t})$', 'agnostic', 'designed'}, 'location', 'northeast', 'Interpreter','latex')
box on
grid minor
set(gcf, 'WindowState', 'maximized');
fontsize(gcf, 20, "points")
xlabel('Median MAPE')
ylabel('Max MAPE')
title('Promisingness of each input data (MAPE)', 'FontSize', 30);
%saveas(gcf, "promisingness_MAPE.png")

figure(); clf();
hold on
yline(agnostic_max_RMSE,'--', 'LineWidth', 3, 'HandleVisibility','off');
xline(agnostic_med_RMSE,'--', 'LineWidth', 3, 'HandleVisibility','off');
plot(linspace(80, 180, 10), linspace(agnostic_med_RMSE + agnostic_max_RMSE - 80, agnostic_med_RMSE + agnostic_max_RMSE - 180, 10), '--r', 'LineWidth', 3, 'HandleVisibility','off')
scatter(cell2mat(X_autoML(ind_A_Q_V, 33)), cell2mat(X_autoML(ind_A_Q_V, 37)), 50, [0 0.4470 0.5410], 'filled')
scatter(cell2mat(X_autoML(ind_A_t_V, 33)), cell2mat(X_autoML(ind_A_t_V, 37)), 50, [0.6500 0.3250 0.0980], 'filled')
scatter(cell2mat(X_autoML(ind_B_Q_V, 33)), cell2mat(X_autoML(ind_B_Q_V, 37)), 70, [0.5 0.5 0.5], 'x', 'LineWidth', 2)
scatter(cell2mat(X_autoML(ind_B_V_t, 33)), cell2mat(X_autoML(ind_B_V_t, 37)), 70, 'g', 'x', 'LineWidth', 2)
scatter(cell2mat(X_autoML(ind_C_Q_V, 33)), cell2mat(X_autoML(ind_C_Q_V, 37)), 50, 'm', '^', 'LineWidth', 1)
scatter(cell2mat(X_autoML(ind_C_V_t, 33)), cell2mat(X_autoML(ind_C_V_t, 37)), 50, 'c', '^', 'LineWidth', 1)
scatter(agnostic_med_RMSE, agnostic_max_RMSE, 800, 'r', 'filled', 'pentagram', 'LineWidth', 1)
scatter(designed_med_RMSE, designed_max_RMSE, 800, 'b', 'filled', 'pentagram', 'LineWidth', 1)
scatter(cell2mat(X_autoML(ind_best_autoML, 33)), cell2mat(X_autoML(ind_best_autoML, 37)), 400, 'k', 'x', 'LineWidth', 4)
xlim([80, 180])
ylim([100, 200])
%legend({'$Q^{\rm{A}}(V)$', '$t^{\rm{A}}(V)$', '$Q^{\rm{B}}(V)$' '$V^{\rm{B}}(\tilde{t})$' '$Q^{\rm{C}}(V)$', '$V^{\rm{C}}(\tilde{t})$', 'agnostic', 'designed'}, 'location', 'northeast', 'Interpreter','latex')
box on
grid minor
set(gcf, 'WindowState', 'maximized');
fontsize(gcf, 20, "points")
xlabel('Median RMSE')
ylabel('Max RMSE')
title('Promisingness of each input data (RMSE)', 'FontSize', 30);
%saveas(gcf, "promisingness_RMSE.png")

figure(); clf();
[ha, pos] = tight_subplot(2,2,[.13 .13],[.08 .03],[.1 .05]);
axes(ha(1));
hold on
yline(agnostic_mean_RMSE,'--', 'LineWidth', 3, 'HandleVisibility','off');
xline(agnostic_mean_MAPE,'--', 'LineWidth', 3, 'HandleVisibility','off');
scatter(cell2mat(X_autoML(ind_B_Q_V, 31)), cell2mat(X_autoML(ind_B_Q_V, 29)), 70, [0.5 0.5 0.5], 'x', 'LineWidth', 2)
scatter(agnostic_mean_MAPE, agnostic_mean_RMSE, 800, 'r', 'filled', 'pentagram', 'LineWidth', 1)
scatter(designed_mean_MAPE, designed_mean_RMSE, 800, 'b', 'filled', 'pentagram', 'LineWidth', 1)
scatter(cell2mat(X_autoML(ind_best_autoML, 31)), cell2mat(X_autoML(ind_best_autoML, 29)), 400, 'k', 'x', 'LineWidth', 4)
xlim([8, 14])
ylim([80, 140])
%legend({'autoML', 'agnostic', 'designed'}, 'location', 'northeast', 'Interpreter','latex')
box on
grid minor
%set(gcf, 'WindowState', 'maximized');
fontsize(gcf, 20, "points")
xlabel('Mean MAPE')
ylabel('Mean RMSE')
%title('Prediction performance (Mean)', 'FontSize', 30);

axes(ha(2));
hold on
yline(agnostic_HL_RMSE,'--', 'LineWidth', 3, 'HandleVisibility','off');
xline(agnostic_HL_MAPE,'--', 'LineWidth', 3, 'HandleVisibility','off');
scatter(cell2mat(X_autoML(ind_B_Q_V, 43)), cell2mat(X_autoML(ind_B_Q_V, 41)), 70, [0.5 0.5 0.5], 'x', 'LineWidth', 2)
scatter(agnostic_HL_MAPE, agnostic_HL_RMSE, 800, 'r', 'filled', 'pentagram', 'LineWidth', 1)
scatter(designed_HL_MAPE, designed_HL_RMSE, 800, 'b', 'filled', 'pentagram', 'LineWidth', 1)
scatter(cell2mat(X_autoML(ind_best_autoML, 43)), cell2mat(X_autoML(ind_best_autoML, 41)), 400, 'k', 'x', 'LineWidth', 4)
xlim([8, 14])
ylim([80, 140])
%legend({'autoML', 'agnostic', 'designed'}, 'location', 'northeast', 'Interpreter','latex')
box on
grid minor
%set(gcf, 'WindowState', 'maximized');
fontsize(gcf, 20, "points")
xlabel('Hodges-Lehmann MAPE')
ylabel('Hodges-Lehmann RMSE')
%title('Prediction performance (Hodges-Lehmann)', 'FontSize', 30);

axes(ha(3));
hold on
yline(agnostic_med_RMSE,'--', 'LineWidth', 3, 'HandleVisibility','off');
xline(agnostic_med_MAPE,'--', 'LineWidth', 3, 'HandleVisibility','off');
scatter(cell2mat(X_autoML(ind_B_Q_V, 35)), cell2mat(X_autoML(ind_B_Q_V, 33)), 70, [0.5 0.5 0.5], 'x', 'LineWidth', 2)
scatter(agnostic_med_MAPE, agnostic_med_RMSE, 800, 'r', 'filled', 'pentagram', 'LineWidth', 1)
scatter(designed_med_MAPE, designed_med_RMSE, 800, 'b', 'filled', 'pentagram', 'LineWidth', 1)
scatter(cell2mat(X_autoML(ind_best_autoML, 35)), cell2mat(X_autoML(ind_best_autoML, 33)), 400, 'k', 'x', 'LineWidth', 4)
xlim([8, 14])
ylim([80, 140])
%legend({'autoML', 'agnostic', 'designed'}, 'location', 'northeast', 'Interpreter','latex')
box on
grid minor
%set(gcf, 'WindowState', 'maximized');
fontsize(gcf, 20, "points")
xlabel('Median MAPE')
ylabel('Median RMSE')
%title('Prediction performance (Median)', 'FontSize', 30);

axes(ha(4));
hold on
yline(agnostic_max_RMSE,'--', 'LineWidth', 3, 'HandleVisibility','off');
xline(agnostic_max_MAPE,'--', 'LineWidth', 3, 'HandleVisibility','off');
scatter(cell2mat(X_autoML(ind_B_Q_V, 39)), cell2mat(X_autoML(ind_B_Q_V, 37)), 70, [0.5 0.5 0.5], 'x', 'LineWidth', 2)
scatter(agnostic_max_MAPE, agnostic_max_RMSE, 800, 'r', 'filled', 'pentagram', 'LineWidth', 1)
scatter(designed_max_MAPE, designed_max_RMSE, 800, 'b', 'filled', 'pentagram', 'LineWidth', 1)
scatter(cell2mat(X_autoML(ind_best_autoML, 39)), cell2mat(X_autoML(ind_best_autoML, 37)), 400, 'k', 'x', 'LineWidth', 4)
xlim([10, 18])
ylim([100, 250])
%legend({'autoML', 'agnostic', 'designed'}, 'location', 'northeast', 'Interpreter','latex')
box on
grid minor
set(gcf, 'WindowState', 'maximized');
fontsize(gcf, 20, "points")
xlabel('Max MAPE')
ylabel('Max RMSE')
%title('Prediction performance (Max)', 'FontSize', 30);


%%
df_prediction = xlsread("prediction_results.xlsx");

y = df_prediction(:, 10);
y_pred_agnostic = df_prediction(:, 11);
y_pred_autoML = df_prediction(:, 12);
y_pred_designed = df_prediction(:, 13);
protocol_index = df_prediction(:, 2);

% Prepare data for boxplot
all_data = [];        % Combined data for boxplot
group_labels = [];    % Grouping for protocol indices
subgroup_labels = []; % Subgroup labels for y and predictions

summary_stats = nan(62, 16);

for i = 1:62
    idx = protocol_index == (i-1);
    if sum(idx) > 0
        % cycle life MAPE
        summary_stats(i, 1) = mean(abs(y(idx) - y(idx)) ./ y(idx));
        summary_stats(i, 2) = mean(abs(y_pred_agnostic(idx) - y(idx)) ./ y(idx));
        summary_stats(i, 3) = mean(abs(y_pred_autoML(idx) - y(idx)) ./ y(idx));
        summary_stats(i, 4) = mean(abs(y_pred_designed(idx) - y(idx)) ./ y(idx));
        % cycle life RMSE
        summary_stats(i, 5) = sqrt(mean((y(idx) - y(idx)).^2));
        summary_stats(i, 6) = sqrt(mean((y_pred_agnostic(idx) - y(idx)).^2));
        summary_stats(i, 7) = sqrt(mean((y_pred_autoML(idx) - y(idx)).^2));
        summary_stats(i, 8) = sqrt(mean((y_pred_designed(idx) - y(idx)).^2));
        % std / mean
        summary_stats(i, 9) = std(y(idx)) / mean(y(idx));
        summary_stats(i, 10) = std(y_pred_agnostic(idx)) / mean(y_pred_agnostic(idx));
        summary_stats(i, 11) = std(y_pred_autoML(idx)) / mean(y_pred_autoML(idx));
        summary_stats(i, 12) = std(y_pred_designed(idx)) / mean(y_pred_designed(idx));
        % Number of samples
        summary_stats(i, 13) = mean(y(idx));
        summary_stats(i, 14) = mean(y_pred_agnostic(idx));
        summary_stats(i, 15) = mean(y_pred_autoML(idx));
        summary_stats(i, 16) = mean(y_pred_designed(idx));
    end
end


%%

ind_fast = [48, 49, 51, 52, 54, 56, 59, 60, 61, 62];
ind_55 = [2, 5, 10, 16, 23, 31, 33, 35, 37];

[~, ind_sort] = sort(summary_stats(:,13));

data = {y, y_pred_agnostic, y_pred_autoML, y_pred_designed};
titles = {'Experimental data', 'Best agnostic model', ...
          'Best autoML model', 'Designed model'};

figure(); clf();
[ha, pos] = tight_subplot(2,2,[.12 .1],[.08 .03],[.08 .05]);
% Create subplots for each data set
for i = 1:4
    % Group data by protocol
    grouped_data = cell(62, 1); % One cell for each protocol index
    y_mean = nan(62, 1);
    y_max = nan(62, 1);
    y_min = nan(62, 1);
    for protocol = 1:62
        grouped_data{protocol} = data{i}(protocol_index == (ind_sort(protocol)-1));
        y_mean(protocol) = mean(grouped_data{protocol});
        y_max(protocol) = max(grouped_data{protocol});
        y_min(protocol) = min(grouped_data{protocol});
    end
    if i == 1
        y_mean_true = y_mean;
        y_max_true = y_max;
        y_min_true = y_min;
        grouped_true_data = grouped_data;
    end

    % Prepare data for boxplot
    boxplot_data = cell2mat(grouped_data); % Concatenate all data
    group_labels = repelem(1:62, cellfun(@numel, grouped_data)); % Protocol indices
    
    ind_fast_rank = nan(length(ind_fast), 1);
    mean_fast_rank = nan(length(ind_fast), 1);
    for k = 1:length(ind_fast)
        ind_fast_rank(k) = find(ind_sort == ind_fast(k));
    end
    ind_55_rank = nan(length(ind_55), 1);
    mean_55_rank = nan(length(ind_55), 1);
    for k = 1:length(ind_55)
        ind_55_rank(k) = find(ind_sort == ind_55(k));
    end
    ind_others = setdiff(1:62, ind_fast_rank);
    ind_others = setdiff(ind_others, ind_55_rank);

    q = quantile(grouped_data{protocol},[0 1]);  
    q0 = q(1);  
    q100 = q(2);  
    
    % Create subplot
    axes(ha(i));
    hold on
    
    ind_odd = find((y_min > y_max_true) | (y_max < y_min_true));
    ind_odd_fast = intersect(ind_odd, ind_fast_rank)';
    ind_odd_55 = intersect(ind_odd, ind_55_rank)';
    ind_odd_others = intersect(ind_odd, ind_others)';
   
    if i > 1
        h = boxplot(cell2mat(grouped_true_data), group_labels, 'Colors', [0.7 0.7 0.7], 'Symbol', 'k.');
        for ii = 1:62
            set(h(5, ii), 'YData', [y_min_true(ii) y_max_true(ii) y_max_true(ii) y_min_true(ii) y_min_true(ii)]);
            upWhisker = get(h(1, i), 'YData');
            set(h(1, ii), 'YData', [upWhisker(2) upWhisker(2)]);
            dwWhisker = get(h(2, ii), 'YData');
            set(h(2, ii), 'YData', [dwWhisker(1) dwWhisker(1)]);
        end
        h = boxplot(boxplot_data, group_labels, 'Colors', 'k', 'Symbol', 'k.');
        for ii = 1:62
            set(h(5, ii), 'YData', [y_min(ii) y_max(ii) y_max(ii) y_min(ii) y_min(ii)]);
            upWhisker = get(h(1, i), 'YData');
            set(h(1, ii), 'YData', [upWhisker(2) upWhisker(2)]);
            dwWhisker = get(h(2, ii), 'YData');
            set(h(2, ii), 'YData', [dwWhisker(1) dwWhisker(1)]);
        end
        scatter(ind_fast_rank, y_mean_true(ind_fast_rank), 50, [0.7 0.7 1], 'filled')
        scatter(ind_55_rank, y_mean_true(ind_55_rank), 50, [1 0.7 0.7], 'filled')
        scatter(ind_others, y_mean_true(ind_others), 50, [0.7 0.7 0.7], 'filled')
        scatter(ind_fast_rank, y_mean(ind_fast_rank), 100, "b", 'filled')
        scatter(ind_55_rank, y_mean(ind_55_rank), 100, "r", 'filled')
        scatter(ind_others, y_mean(ind_others), 100, "k", 'filled')
    else
        h = boxplot(boxplot_data, group_labels, 'Colors', 'k', 'Symbol', 'k.');
        for ii = 1:62
            set(h(5, ii), 'YData', [y_min(ii) y_max(ii) y_max(ii) y_min(ii) y_min(ii)]);
            upWhisker = get(h(1, i), 'YData');
            set(h(1, ii), 'YData', [upWhisker(2) upWhisker(2)]);
            dwWhisker = get(h(2, ii), 'YData');
            set(h(2, ii), 'YData', [dwWhisker(1) dwWhisker(1)]);
        end
        scatter(ind_fast_rank, y_mean_true(ind_fast_rank), 100, "b", 'filled')
        scatter(ind_55_rank, y_mean_true(ind_55_rank), 100, "r", 'filled')
        scatter(ind_others, y_mean_true(ind_others), 100, "k", 'filled')
    end

    %title(titles{i});
    xlabel('Protocol ranked by cycle life');
    ylabel('Cycle life');
    set(gca,'XGrid','off','YGrid','on')
    set(gca,'XTickLabelMode','auto')
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',20);
    set(gca,'xticklabel',{[]})
    ylim([400, 1400])

    if i > 1
        switch i
            case 2
                axes('Position',[.65 .8 .15 .15])
            case 3
                axes('Position',[.16 .3 .15 .15])
            case 4
                axes('Position',[.65 .3 .15 .15])
        end
        edges = linspace(0, 350, 36);
        box on
        histogram(abs(y_mean - y_mean_true), edges, 'FaceColor', [0.5, 0.5, 0.5])
        hold on
        xline(250, '--k')
        xlim([0 400])
        ylim([0, 15])


    end

    set(gcf, 'WindowState', 'maximized');
    
end

%%
figure(); clf();
[ha, pos] = tight_subplot(2,1,[.12 .1],[.08 .03],[.08 .05]);
% Create subplots for each data set
i_list = [1, 4];
for kk = 1:length(i_list)
    i = i_list(kk);
    % Group data by protocol
    grouped_data = cell(62, 1); % One cell for each protocol index
    y_mean = nan(62, 1);
    y_max = nan(62, 1);
    y_min = nan(62, 1);
    for protocol = 1:62
        grouped_data{protocol} = data{i}(protocol_index == (ind_sort(protocol)-1));
        y_mean(protocol) = mean(grouped_data{protocol});
        y_max(protocol) = max(grouped_data{protocol});
        y_min(protocol) = min(grouped_data{protocol});
    end
    if i == 1
        y_mean_true = y_mean;
        y_max_true = y_max;
        y_min_true = y_min;
        grouped_true_data = grouped_data;
    end

    % Prepare data for boxplot
    boxplot_data = cell2mat(grouped_data); % Concatenate all data
    group_labels = repelem(1:62, cellfun(@numel, grouped_data)); % Protocol indices
    
    ind_fast_rank = nan(length(ind_fast), 1);
    mean_fast_rank = nan(length(ind_fast), 1);
    for k = 1:length(ind_fast)
        ind_fast_rank(k) = find(ind_sort == ind_fast(k));
    end
    ind_55_rank = nan(length(ind_55), 1);
    mean_55_rank = nan(length(ind_55), 1);
    for k = 1:length(ind_55)
        ind_55_rank(k) = find(ind_sort == ind_55(k));
    end
    ind_others = setdiff(1:62, ind_fast_rank);
    ind_others = setdiff(ind_others, ind_55_rank);

    q = quantile(grouped_data{protocol},[0 1]);  
    q0 = q(1);  
    q100 = q(2);  
    
    % Create subplot
    axes(ha(kk));
    hold on
    
    ind_odd = find((y_min > y_max_true) | (y_max < y_min_true));
    ind_odd_fast = intersect(ind_odd, ind_fast_rank)';
    ind_odd_55 = intersect(ind_odd, ind_55_rank)';
    ind_odd_others = intersect(ind_odd, ind_others)';
   
    if i > 1
        h = boxplot(cell2mat(grouped_true_data), group_labels, 'Colors', [0.7 0.7 0.7], 'Symbol', 'k.');
        for ii = 1:62
            set(h(5, ii), 'YData', [y_min_true(ii) y_max_true(ii) y_max_true(ii) y_min_true(ii) y_min_true(ii)]);
            upWhisker = get(h(1, i), 'YData');
            set(h(1, ii), 'YData', [upWhisker(2) upWhisker(2)]);
            dwWhisker = get(h(2, ii), 'YData');
            set(h(2, ii), 'YData', [dwWhisker(1) dwWhisker(1)]);
        end
        h = boxplot(boxplot_data, group_labels, 'Colors', 'k', 'Symbol', 'k.');
        for ii = 1:62
            set(h(5, ii), 'YData', [y_min(ii) y_max(ii) y_max(ii) y_min(ii) y_min(ii)]);
            upWhisker = get(h(1, i), 'YData');
            set(h(1, ii), 'YData', [upWhisker(2) upWhisker(2)]);
            dwWhisker = get(h(2, ii), 'YData');
            set(h(2, ii), 'YData', [dwWhisker(1) dwWhisker(1)]);
        end
        scatter(ind_fast_rank, y_mean_true(ind_fast_rank), 50, [0.7 0.7 1], 'filled')
        scatter(ind_55_rank, y_mean_true(ind_55_rank), 50, [1 0.7 0.7], 'filled')
        scatter(ind_others, y_mean_true(ind_others), 50, [0.7 0.7 0.7], 'filled')
        scatter(ind_fast_rank, y_mean(ind_fast_rank), 100, "b", 'filled')
        scatter(ind_55_rank, y_mean(ind_55_rank), 100, "r", 'filled')
        scatter(ind_others, y_mean(ind_others), 100, "k", 'filled')
    else
        h = boxplot(boxplot_data, group_labels, 'Colors', 'k', 'Symbol', 'k.');
        for ii = 1:62
            set(h(5, ii), 'YData', [y_min(ii) y_max(ii) y_max(ii) y_min(ii) y_min(ii)]);
            upWhisker = get(h(1, i), 'YData');
            set(h(1, ii), 'YData', [upWhisker(2) upWhisker(2)]);
            dwWhisker = get(h(2, ii), 'YData');
            set(h(2, ii), 'YData', [dwWhisker(1) dwWhisker(1)]);
        end
        scatter(ind_fast_rank, y_mean_true(ind_fast_rank), 100, "b", 'filled')
        scatter(ind_55_rank, y_mean_true(ind_55_rank), 100, "r", 'filled')
        scatter(ind_others, y_mean_true(ind_others), 100, "k", 'filled')
    end

    title(titles{i});
    xlabel('Protocol ranked by cycle life');
    ylabel('Cycle life');
    set(gca,'XGrid','off','YGrid','on')
    set(gca,'XTickLabelMode','auto')
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',20);
    set(gca,'xticklabel',{[]})
    ylim([400, 1400])

    if kk > 1
       
        axes('Position',[.17 .3 .3 .13])
        
        edges = linspace(0, 350, 36);
        box on
        histogram(abs(y_mean - y_mean_true), edges, 'FaceColor', [0.5, 0.5, 0.5])
        hold on
        xline(250, '--k')
        xlim([0 400])
        ylim([0, 15])

    end

    set(gcf, 'WindowState', 'maximized');
    
end

%% Performance of R_LS feature

fontsize = 16;
df_R_LS = xlsread("R_LS_formation_dataset.xlsx");
id_isnan = logical(~isnan(df_R_LS(:, 11)) .* ~isnan(df_R_LS(:, 12)) .* ~isnan(df_R_LS(:, 13)));
T = df_R_LS(id_isnan, 5);
SoC = df_R_LS(id_isnan, 11);
R_LS = df_R_LS(id_isnan, 12);
cl = df_R_LS(id_isnan, 13);

figure(); clf();
[ha, pos] = tight_subplot(3,3,[.12 .05],[.08 .05],[.08 .05]);
SoC_list = [5, 7, 8, 9, 10, 11, 9, 10, 11.1];
cmap_T_total = cool(9);
for k=1:6
    axes(ha(k));
    id_SoC = abs(SoC-SoC_list(k)) < 0.25;
    [~,ddd] = sort(T(id_SoC));
    ind = find(id_SoC);
    cmap_T = cmap_T_total(unique(T(id_SoC))/5-4, :);
    gscatter(R_LS(ind(ddd)), cl(ind(ddd)), T(ind(ddd)), cmap_T,'o', 10, "filled")
    hold on
    % use handle for plotting
    xbar = mean(R_LS(id_SoC));
    ybar = mean(cl(id_SoC));
    beta = sum((R_LS(id_SoC) - xbar) .* (cl(id_SoC) - ybar)) / sum((R_LS(id_SoC) - xbar).^2);
    alpha = ybar - beta * xbar;
    xgrid = linspace(min(R_LS(id_SoC)), max(R_LS(id_SoC)), 100);
    R = corrcoef(R_LS(id_SoC), cl(id_SoC));
    plot(xgrid, alpha + beta * xgrid, '--k', 'LineWidth', 2, 'HandleVisibility','off');
    fig_title = num2str(round(SoC_list(k) - 0.25, 2)) + "% < SoC < " + num2str(round(SoC_list(k) + 0.25, 2)) + "%" + "  (\rho = " + num2str(round(R(1,2),2)) + ")";
    xlim([min(R_LS(id_SoC))-0.05, max(R_LS(id_SoC))+0.05])
    ylim([min(cl(id_SoC))-50, max(cl(id_SoC))+50])
    xlabel("R_{LS}", 'FontSize', fontsize);
    ylabel("Cycle life", 'FontSize', fontsize);
    grid minor
    box on
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',12)
    title(fig_title, 'FontSize', fontsize)
    lgd = legend('show', 'location', 'northeast');
    set(lgd, 'FontSize',16)
end
for k=7:9
    axes(ha(k));
    id_SoC = abs(SoC-SoC_list(k)) < 0.1;
    [~,ddd] = sort(T(id_SoC));
    ind = find(id_SoC);
    cmap_T = cmap_T_total(unique(T(id_SoC))/5-4, :);
    gscatter(R_LS(ind(ddd)), cl(ind(ddd)), T(ind(ddd)), cmap_T,'o', 10, "filled")
    hold on
    % use handle for plotting
    xbar = mean(R_LS(id_SoC));
    ybar = mean(cl(id_SoC));
    beta = sum((R_LS(id_SoC) - xbar) .* (cl(id_SoC) - ybar)) / sum((R_LS(id_SoC) - xbar).^2);
    alpha = ybar - beta * xbar;
    xgrid = linspace(min(R_LS(id_SoC)), max(R_LS(id_SoC)), 100);
    R = corrcoef(R_LS(id_SoC), cl(id_SoC));
    plot(xgrid, alpha + beta * xgrid, '--k', 'LineWidth', 2, 'HandleVisibility','off');
    fig_title = num2str(round(SoC_list(k) - 0.1, 2)) + "% < SoC < " + num2str(round(SoC_list(k) + 0.1, 2)) + "%" + "  (\rho = " + num2str(round(R(1,2),2)) + ")";
    xlim([min(R_LS(id_SoC))-0.05, max(R_LS(id_SoC))+0.05])
    ylim([min(cl(id_SoC))-50, max(cl(id_SoC))+50])
    xlabel("R_{LS}", 'FontSize', fontsize);
    ylabel("Cycle life", 'FontSize', fontsize);
    grid minor
    box on
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',12)
    title(fig_title, 'FontSize', fontsize)
    lgd = legend('show', 'location', 'northeast');
    set(lgd, 'FontSize',16)
end
set(gcf, 'WindowState', 'maximized');

