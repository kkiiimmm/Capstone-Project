function fullsearch_power()
% Full-search transmit power labeling for a multi-router IoT network.
%
% H(k,m) is the channel gain from router m to device k.
% H is already a power gain/pathloss-scale value, so it is used directly.
% Do not square H in SINR or capacity calculations.

    clear;
    clc;

    K = 6;
    POWER_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0];
    NOISE_POWER = 1e-6;
    DEBUG_MODE = false;

    TRAIN_INPUT_FILE = 'data/H_train.dat';
    TEST_INPUT_FILE = 'data/H_test.dat';
    TRAIN_OUTPUT_FILE = 'data/label_train.dat';
    TEST_OUTPUT_FILE = 'data/label_test.dat';

    process_dataset(TRAIN_INPUT_FILE, TRAIN_OUTPUT_FILE, K, POWER_CANDIDATES, NOISE_POWER, DEBUG_MODE);
    process_dataset(TEST_INPUT_FILE, TEST_OUTPUT_FILE, K, POWER_CANDIDATES, NOISE_POWER, DEBUG_MODE);
end


function sinr = compute_sinr(H, P, noise_power)
    K = length(P);
    sinr = zeros(1, K);

    for k = 1:K
        % Desired signal from router k to its paired IoT device k.
        desired_signal = P(k) * H(k, k);

        % Interference at device k from every other router m ~= k.
        interference = 0.0;
        for m = 1:K
            if m ~= k
                interference = interference + P(m) * H(k, m);
            end
        end

        % H is a power gain/pathloss-scale value, not an amplitude.
        % Therefore the SINR uses H directly, without squaring H.
        sinr(k) = desired_signal / (noise_power + interference);
    end

    if any(sinr < 0)
        error('SINR contains negative values.');
    end
end


function capacity = compute_sum_capacity(H, P, noise_power)
    sinr = compute_sinr(H, P, noise_power);
    capacity = sum(log2(1.0 + sinr));

    if ~isfinite(capacity)
        error('Capacity is not finite.');
    end
end


function [best_capacity, best_power] = fullsearch_one_sample(H, power_candidates, noise_power)
    K = size(H, 1);
    persistent cached_power_candidates cached_K cached_power_vectors;

    if isempty(cached_power_vectors) || cached_K ~= K || ...
            ~isequal(cached_power_candidates, power_candidates)
        cached_power_candidates = power_candidates;
        cached_K = K;
        cached_power_vectors = generate_power_vectors(power_candidates, K);
    end

    power_vectors = cached_power_vectors;

    best_capacity = -Inf;
    best_power = zeros(1, K);

    for i = 1:size(power_vectors, 1)
        P = power_vectors(i, :);
        capacity = compute_sum_capacity(H, P, noise_power);

        if capacity > best_capacity
            best_capacity = capacity;
            best_power = P;
        end
    end
end


function process_dataset(input_file, output_file, K, power_candidates, noise_power, debug_mode)
    H_rows = load(input_file);

    if size(H_rows, 2) ~= K * K
        error('Each H row must have K*K elements. Got %d columns.', size(H_rows, 2));
    end

    sample_count = size(H_rows, 1);
    if debug_mode
        if ~isempty(strfind(input_file, 'train'))
            sample_count = min(sample_count, 100);
        else
            sample_count = min(sample_count, 20);
        end
    end

    labels = zeros(sample_count, K + 1);

    fprintf('Processing %s: %d samples\n', input_file, sample_count);

    for n = 1:sample_count
        row = H_rows(n, :);

        if numel(row) ~= K * K
            error('H row %d must have K*K elements.', n);
        end

        % NumPy saved H using row-major flattening:
        % H11 H12 ... H1K H21 H22 ... HKK.
        % MATLAB reshape is column-major, so transpose after reshape.
        H = reshape(row, [K, K])';

        [best_capacity, best_power] = fullsearch_one_sample(H, power_candidates, noise_power);
        labels(n, :) = [best_capacity, best_power];
    end

    if size(labels, 2) ~= K + 1
        error('Label output must have K+1 columns.');
    end

    save(output_file, 'labels', '-ascii');
    fprintf('Saved %s\n', output_file);
end


function power_vectors = generate_power_vectors(power_candidates, K)
    candidate_count = length(power_candidates);
    combination_count = candidate_count ^ K;
    power_vectors = zeros(combination_count, K);

    for index = 0:(combination_count - 1)
        value = index;
        for k = 1:K
            candidate_index = mod(value, candidate_count) + 1;
            power_vectors(index + 1, k) = power_candidates(candidate_index);
            value = floor(value / candidate_count);
        end
    end
end
