% Visualization script for the graduation project.
%
% This file is intentionally written as a script for MATLAB R2014a.
% MATLAB R2014a does not support local functions in scripts, so helper logic
% is kept in clear script blocks.
%
% H(k,m) is the channel gain from router m to device k.
% H is already a power gain/pathloss-scale value, so H is not squared.

clear;
clc;

K = 6;
AREA_SIZE = 100;
TRAIN_SAMPLE_INDEX = 1;
TEST_SAMPLE_INDEX = 1;
NUM_INTERFERENCE_LINKS = 8;
NOISE_POWER = 1e-6;
EPSILON = 1e-30;
ON_THRESHOLD = 1e-5;
POWER_CANDIDATES = [0.25, 0.5, 0.75, 1.0];
RANDOM_SEED = 42;

if exist('figures', 'dir') ~= 7
    mkdir('figures');
end

positions_train = load('data/positions_train.dat');
positions_test = load('data/positions_test.dat');
H_train_rows = load('data/H_train.dat');
H_test_rows = load('data/H_test.dat');
label_test = load('data/label_test.dat');
dnn_result = load('results/dnn_result.dat');
unsup_result = load('results/unsupervised_result.dat');
dqn_result = load('results/dqn_result.dat');

if size(positions_train, 2) ~= 4 * K || size(positions_test, 2) ~= 4 * K
    error('Position files must have 24 columns for K = 6.');
end
if size(H_train_rows, 2) ~= K * K || size(H_test_rows, 2) ~= K * K
    error('H files must have 36 columns for K = 6.');
end
if size(label_test, 2) ~= K + 1
    error('label_test.dat must have 7 columns.');
end
if size(dnn_result, 2) ~= 10
    error('dnn_result.dat must have 10 columns.');
end
if size(unsup_result, 2) ~= 18
    error('unsupervised_result.dat must have 18 columns.');
end
if size(dqn_result, 2) ~= 11
    error('dqn_result.dat must have 11 columns.');
end

sample_count = min([size(H_test_rows, 1), size(label_test, 1), size(dnn_result, 1), ...
    size(unsup_result, 1), size(dqn_result, 1)]);

% Parse representative train positions.
train_row = positions_train(TRAIN_SAMPLE_INDEX, :);
train_router_pos = reshape(train_row(1:2*K), [2, K])';
train_device_pos = reshape(train_row(2*K+1:4*K), [2, K])';

% Parse representative test positions and H.
test_row = positions_test(TEST_SAMPLE_INDEX, :);
test_router_pos = reshape(test_row(1:2*K), [2, K])';
test_device_pos = reshape(test_row(2*K+1:4*K), [2, K])';

H_test_sample = reshape(H_test_rows(TEST_SAMPLE_INDEX, :), [K, K])';

font_size = 13;
line_width = 1.5;

% -------------------------------------------------------------------------
% 1. Training sample network layout
% -------------------------------------------------------------------------
figure('Color', 'w', 'Visible', 'off', 'Toolbar', 'none', 'Menubar', 'none');
set(gca, 'NextPlot', 'add');
for k = 1:K
    plot([train_router_pos(k, 1), train_device_pos(k, 1)], ...
        [train_router_pos(k, 2), train_device_pos(k, 2)], 'k-', 'LineWidth', line_width);
end
plot(train_router_pos(:, 1), train_router_pos(:, 2), 'r^', ...
    'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(train_device_pos(:, 1), train_device_pos(:, 2), 'bo', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b');
for k = 1:K
    text(train_router_pos(k, 1) + 1.5, train_router_pos(k, 2) + 1.5, ...
        sprintf('R%d', k), 'FontSize', font_size);
    text(train_device_pos(k, 1) + 1.5, train_device_pos(k, 2) + 1.5, ...
        sprintf('D%d', k), 'FontSize', font_size);
end
xlim([0 AREA_SIZE]);
ylim([0 AREA_SIZE]);
axis square;
grid on;
xlabel('x position (m)', 'FontSize', font_size);
ylabel('y position (m)', 'FontSize', font_size);
title('Training Sample Network Layout', 'FontSize', font_size + 2);
legend('Desired link', 'Router', 'IoT device', 'Location', 'best');
set(gca, 'FontSize', font_size);
print(gcf, '-dpng', '-r300', 'figures/network_train_sample.png');

% -------------------------------------------------------------------------
% 2. Test sample network layout
% -------------------------------------------------------------------------
figure('Color', 'w', 'Visible', 'off', 'Toolbar', 'none', 'Menubar', 'none');
set(gca, 'NextPlot', 'add');
for k = 1:K
    plot([test_router_pos(k, 1), test_device_pos(k, 1)], ...
        [test_router_pos(k, 2), test_device_pos(k, 2)], 'k-', 'LineWidth', line_width);
end
plot(test_router_pos(:, 1), test_router_pos(:, 2), 'r^', ...
    'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(test_device_pos(:, 1), test_device_pos(:, 2), 'bo', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b');
for k = 1:K
    text(test_router_pos(k, 1) + 1.5, test_router_pos(k, 2) + 1.5, ...
        sprintf('R%d', k), 'FontSize', font_size);
    text(test_device_pos(k, 1) + 1.5, test_device_pos(k, 2) + 1.5, ...
        sprintf('D%d', k), 'FontSize', font_size);
end
xlim([0 AREA_SIZE]);
ylim([0 AREA_SIZE]);
axis square;
grid on;
xlabel('x position (m)', 'FontSize', font_size);
ylabel('y position (m)', 'FontSize', font_size);
title('Test Sample Network Layout', 'FontSize', font_size + 2);
legend('Desired link', 'Router', 'IoT device', 'Location', 'best');
set(gca, 'FontSize', font_size);
print(gcf, '-dpng', '-r300', 'figures/network_test_sample.png');

% -------------------------------------------------------------------------
% 3. Channel gain heatmap
% -------------------------------------------------------------------------
figure('Color', 'w', 'Visible', 'off', 'Toolbar', 'none', 'Menubar', 'none');
imagesc(log10(H_test_sample + EPSILON));
axis square;
colorbar;
set(gca, 'NextPlot', 'add');
for k = 1:K
    plot(k, k, 'ws', 'MarkerSize', 14, 'LineWidth', 2);
end
xlabel('Router index m', 'FontSize', font_size);
ylabel('Device index k', 'FontSize', font_size);
title('Channel Gain Matrix H', 'FontSize', font_size + 2);
set(gca, 'XTick', 1:K, 'YTick', 1:K, 'FontSize', font_size);
print(gcf, '-dpng', '-r300', 'figures/test_channel_heatmap.png');

% -------------------------------------------------------------------------
% 4. Desired links and strongest interference links
% -------------------------------------------------------------------------
figure('Color', 'w', 'Visible', 'off', 'Toolbar', 'none', 'Menubar', 'none');
set(gca, 'NextPlot', 'add');
for k = 1:K
    plot([test_router_pos(k, 1), test_device_pos(k, 1)], ...
        [test_router_pos(k, 2), test_device_pos(k, 2)], 'k-', 'LineWidth', 2);
end

interference_values = [];
interference_device = [];
interference_router = [];
for k = 1:K
    for m = 1:K
        if m ~= k
            interference_values = [interference_values; H_test_sample(k, m)];
            interference_device = [interference_device; k];
            interference_router = [interference_router; m];
        end
    end
end
[sorted_values, order] = sort(interference_values, 'descend');
draw_count = min(NUM_INTERFERENCE_LINKS, length(order));
for idx = 1:draw_count
    k = interference_device(order(idx));
    m = interference_router(order(idx));
    plot([test_router_pos(m, 1), test_device_pos(k, 1)], ...
        [test_router_pos(m, 2), test_device_pos(k, 2)], 'Color', [0.5 0.5 0.5], ...
        'LineStyle', '--', 'LineWidth', 1.2);
end
plot(test_router_pos(:, 1), test_router_pos(:, 2), 'r^', ...
    'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(test_device_pos(:, 1), test_device_pos(:, 2), 'bo', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'b');
for k = 1:K
    text(test_router_pos(k, 1) + 1.5, test_router_pos(k, 2) + 1.5, ...
        sprintf('R%d', k), 'FontSize', font_size);
    text(test_device_pos(k, 1) + 1.5, test_device_pos(k, 2) + 1.5, ...
        sprintf('D%d', k), 'FontSize', font_size);
end
xlim([0 AREA_SIZE]);
ylim([0 AREA_SIZE]);
axis square;
grid on;
xlabel('x position (m)', 'FontSize', font_size);
ylabel('y position (m)', 'FontSize', font_size);
title('Desired Links and Strong Interference Links', 'FontSize', font_size + 2);
legend('Desired link', 'Strong interference', 'Router', 'IoT device', 'Location', 'best');
set(gca, 'FontSize', font_size);
print(gcf, '-dpng', '-r300', 'figures/test_interference_graph.png');

% -------------------------------------------------------------------------
% 5. Power allocation for one test sample
% -------------------------------------------------------------------------
full_power = label_test(TEST_SAMPLE_INDEX, 2:K+1);
dnn_power = dnn_result(TEST_SAMPLE_INDEX, 5:10);
unsup_hard_power = unsup_result(TEST_SAMPLE_INDEX, 13:18);
dqn_power = dqn_result(TEST_SAMPLE_INDEX, 6:11);

power_matrix = [full_power; dnn_power; unsup_hard_power; dqn_power]';

figure('Color', 'w', 'Visible', 'off', 'Toolbar', 'none', 'Menubar', 'none');
bar(1:K, power_matrix, 'grouped');
grid on;
xlabel('Router index', 'FontSize', font_size);
ylabel('Transmit power', 'FontSize', font_size);
title('Power Allocation for One Test Sample', 'FontSize', font_size + 2);
legend('Full-search', 'Supervised DNN', 'Unsupervised hard', 'DQN', 'Location', 'best');
set(gca, 'XTick', 1:K, 'FontSize', font_size);
print(gcf, '-dpng', '-r300', 'figures/power_allocation_sample.png');

% -------------------------------------------------------------------------
% 6. Average capacity ratio and baseline ratios
% -------------------------------------------------------------------------
H_test_used = H_test_rows(1:sample_count, :);
full_capacity = label_test(1:sample_count, 1);
dnn_capacity = dnn_result(1:sample_count, 3);
unsup_hard_capacity = unsup_result(1:sample_count, 5);
dqn_capacity = dqn_result(1:sample_count, 3);

rand('seed', RANDOM_SEED);
random_capacity = zeros(sample_count, 1);
allmax_capacity = zeros(sample_count, 1);
allmax_power = 0.99 * ones(1, K);
for i = 1:sample_count
    H_i = reshape(H_test_used(i, :), [K, K])';
    random_indices = floor(rand(1, K) * length(POWER_CANDIDATES)) + 1;
    random_power = POWER_CANDIDATES(random_indices);

    % Random baseline capacity.
    desired_signal = random_power .* diag(H_i)';
    received_power = H_i * random_power';
    interference = received_power' - desired_signal;
    sinr = desired_signal ./ (NOISE_POWER + interference);
    if any(sinr < 0)
        error('SINR contains negative values.');
    end
    random_capacity(i) = sum(log2(1 + sinr));

    % All-max baseline capacity.
    desired_signal = allmax_power .* diag(H_i)';
    received_power = H_i * allmax_power';
    interference = received_power' - desired_signal;
    sinr = desired_signal ./ (NOISE_POWER + interference);
    if any(sinr < 0)
        error('SINR contains negative values.');
    end
    allmax_capacity(i) = sum(log2(1 + sinr));
end
if any(~isfinite(random_capacity)) || any(~isfinite(allmax_capacity))
    error('Baseline capacity contains non-finite values.');
end

full_ratio = ones(sample_count, 1);
dnn_ratio = dnn_capacity ./ full_capacity;
unsup_hard_ratio = unsup_hard_capacity ./ full_capacity;
dqn_ratio = dqn_capacity ./ full_capacity;
random_ratio = random_capacity ./ full_capacity;
allmax_ratio = allmax_capacity ./ full_capacity;

avg_ratios = [mean(full_ratio), mean(dnn_ratio), mean(unsup_hard_ratio), ...
    mean(dqn_ratio), mean(random_ratio), mean(allmax_ratio)];

figure('Color', 'w', 'Visible', 'off', 'Toolbar', 'none', 'Menubar', 'none');
bar(avg_ratios);
grid on;
ylabel('Average capacity ratio', 'FontSize', font_size);
title('Average Capacity Ratio', 'FontSize', font_size + 2);
set(gca, 'XTick', 1:6, 'XTickLabel', {'Full', 'DNN', 'Unsup', 'DQN', 'Random', 'All-max'}, ...
    'FontSize', font_size);
ylim([0, 1.1]);
print(gcf, '-dpng', '-r300', 'figures/average_capacity_ratio.png');

% -------------------------------------------------------------------------
% 7. Capacity ratio boxplot
% -------------------------------------------------------------------------
figure('Color', 'w', 'Visible', 'off', 'Toolbar', 'none', 'Menubar', 'none');
boxplot([dnn_ratio, unsup_hard_ratio, dqn_ratio, random_ratio, allmax_ratio], ...
    'Labels', {'DNN', 'Unsup', 'DQN', 'Random', 'All-max'});
grid on;
ylabel('Capacity ratio', 'FontSize', font_size);
title('Capacity Ratio Distribution', 'FontSize', font_size + 2);
set(gca, 'FontSize', font_size);
print(gcf, '-dpng', '-r300', 'figures/capacity_ratio_boxplot.png');

% -------------------------------------------------------------------------
% 8. Router activation ratio
% -------------------------------------------------------------------------
full_powers = label_test(1:sample_count, 2:K+1);
dnn_powers = dnn_result(1:sample_count, 5:10);
unsup_hard_powers = unsup_result(1:sample_count, 13:18);
dqn_powers = dqn_result(1:sample_count, 6:11);

on_ratios = [mean(full_powers > ON_THRESHOLD); ...
    mean(dnn_powers > ON_THRESHOLD); ...
    mean(unsup_hard_powers > ON_THRESHOLD); ...
    mean(dqn_powers > ON_THRESHOLD)]';

figure('Color', 'w', 'Visible', 'off', 'Toolbar', 'none', 'Menubar', 'none');
bar(1:K, on_ratios, 'grouped');
grid on;
xlabel('Router index', 'FontSize', font_size);
ylabel('ON ratio', 'FontSize', font_size);
title('Router Activation Ratio', 'FontSize', font_size + 2);
legend('Full-search', 'Supervised DNN', 'Unsupervised hard', 'DQN', 'Location', 'best');
set(gca, 'XTick', 1:K, 'FontSize', font_size);
ylim([0, 1.05]);
print(gcf, '-dpng', '-r300', 'figures/router_on_ratio.png');

% -------------------------------------------------------------------------
% 9. DQN action distribution
% -------------------------------------------------------------------------
action_indices = dqn_result(1:sample_count, 5);
unique_actions = unique(action_indices);
action_counts = zeros(length(unique_actions), 1);
for i = 1:length(unique_actions)
    action_counts(i) = sum(action_indices == unique_actions(i));
end
[sorted_counts, order] = sort(action_counts, 'descend');
top_count = min(10, length(order));
top_actions = unique_actions(order(1:top_count));
top_counts = sorted_counts(1:top_count);

figure('Color', 'w', 'Visible', 'off', 'Toolbar', 'none', 'Menubar', 'none');
bar(top_counts);
grid on;
xlabel('Action index', 'FontSize', font_size);
ylabel('Selection count', 'FontSize', font_size);
title('Top DQN Actions', 'FontSize', font_size + 2);
set(gca, 'XTick', 1:top_count, 'XTickLabel', cellstr(num2str(top_actions)), ...
    'FontSize', font_size);
print(gcf, '-dpng', '-r300', 'figures/dqn_action_distribution.png');

fprintf('Saved visualization figures to figures/ directory.\n');
