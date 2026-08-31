%% ============================================================
% Generate 10 Random Training–Testing Dataset Splits
%
% Each split exports:
%   1) Training dataset (75%) + TRAIN_DISTRIBUTION
%   2) Testing dataset  (25%) + TEST_DISTRIBUTION
%
% Selected variables:
%   D, t, Fy, Fc, L_D, N
%
% Statistics:
%   N, Min, Max, Mean, Median, SD, COV(%)
% ============================================================

clear; clc; close all;

%% ===================== CONFIG =====================

input_file = "DATASET_N_S.xlsx";

train_ratio = 0.75;

number_of_sets = 10;

initial_random_state = 32;

output_folder = "Random_Dataset_Splits";

%% ===================== CREATE OUTPUT FOLDER =====================

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% ===================== LOAD DATA =====================

df = readtable(input_file, 'VariableNamingRule', 'preserve');

fprintf('\n====================================================\n');
fprintf('ORIGINAL DATASET\n');
fprintf('====================================================\n');
fprintf('Rows    : %d\n', height(df));
fprintf('Columns : %d\n', width(df));

%% ===================== RENAME VARIABLES =====================

variableNames = df.Properties.VariableNames;

% Rename L/D to L_D
if any(strcmp(variableNames, 'L/D'))
    df.Properties.VariableNames{strcmp(variableNames, 'L/D')} = 'L_D';
end

% รองรับกรณีชื่อคอลัมน์เป็น fy และ fc
if any(strcmp(df.Properties.VariableNames, 'fy'))
    df.Properties.VariableNames{strcmp( ...
        df.Properties.VariableNames, 'fy')} = 'Fy';
end

if any(strcmp(df.Properties.VariableNames, 'fc'))
    df.Properties.VariableNames{strcmp( ...
        df.Properties.VariableNames, 'fc')} = 'Fc';
end

%% ===================== KEEP REQUIRED COLUMNS =====================

requiredVars = {'D', 't', 'Fy', 'Fc', 'L_D', 'N'};

missingVars = setdiff(requiredVars, df.Properties.VariableNames);

if ~isempty(missingVars)
    error('Missing required variables: %s', ...
        strjoin(missingVars, ', '));
end

df = df(:, requiredVars);

%% ===================== CHECK MISSING VALUES =====================

missingRows = any(ismissing(df), 2);

if any(missingRows)
    fprintf('\nWarning: Removing %d rows containing missing values.\n', ...
        sum(missingRows));

    df(missingRows, :) = [];
end

%% ===================== SPLIT INFORMATION =====================

n_total = height(df);
n_train = floor(train_ratio * n_total);
n_test  = n_total - n_train;

fprintf('\n====================================================\n');
fprintf('SPLIT INFORMATION\n');
fprintf('====================================================\n');
fprintf('Number of sets : %d\n', number_of_sets);
fprintf('Training rows  : %d\n', n_train);
fprintf('Testing rows   : %d\n', n_test);
fprintf('Train ratio    : %.2f%%\n', 100 * n_train / n_total);
fprintf('Test ratio     : %.2f%%\n', 100 * n_test / n_total);

%% ===================== GENERATE 10 DATASET SETS =====================

for setNumber = 1:number_of_sets

    random_state = initial_random_state + setNumber - 1;

    %% ----------------- RANDOM SHUFFLE -----------------

    rng(random_state, 'twister');

    shuffledIndex = randperm(n_total);

    df_shuffled = df(shuffledIndex, :);

    %% ----------------- TRAIN–TEST SPLIT -----------------

    df_train = df_shuffled(1:n_train, :);

    df_test = df_shuffled(n_train + 1:end, :);

    %% ----------------- CREATE SUMMARIES -----------------

    summary_train = createDistributionSummary( ...
        df_train, requiredVars);

    summary_test = createDistributionSummary( ...
        df_test, requiredVars);

    %% ----------------- OUTPUT FILENAMES -----------------

    trainFilename = fullfile( ...
        output_folder, ...
        sprintf('N_S_Training_DATASET_75_Set_%02d.xlsx', ...
        setNumber));

    testFilename = fullfile( ...
        output_folder, ...
        sprintf('N_S_Testing_DATASET_25_Set_%02d.xlsx', ...
        setNumber));

    %% ----------------- REMOVE OLD FILES -----------------

    if isfile(trainFilename)
        delete(trainFilename);
    end

    if isfile(testFilename)
        delete(testFilename);
    end

    %% ----------------- EXPORT TRAINING FILE -----------------

    writetable(df_train, trainFilename, ...
        'Sheet', 'TRAIN_DATA');

    writetable(summary_train, trainFilename, ...
        'Sheet', 'TRAIN_DISTRIBUTION');

    %% ----------------- EXPORT TESTING FILE -----------------

    writetable(df_test, testFilename, ...
        'Sheet', 'TEST_DATA');

    writetable(summary_test, testFilename, ...
        'Sheet', 'TEST_DISTRIBUTION');

    %% ----------------- PRINT PROGRESS -----------------

    fprintf('\nSet %02d completed | Seed = %d\n', ...
        setNumber, random_state);

    fprintf('  Training: %d rows\n', height(df_train));
    fprintf('  Testing : %d rows\n', height(df_test));

end

%% ===================== COMPLETION MESSAGE =====================

fprintf('\n====================================================\n');
fprintf('EXPORT COMPLETED SUCCESSFULLY!\n');
fprintf('====================================================\n');

fprintf('Training files created : %d\n', number_of_sets);
fprintf('Testing files created  : %d\n', number_of_sets);
fprintf('Total files created    : %d\n', 2 * number_of_sets);
fprintf('Output folder          : %s\n', output_folder);

%% ============================================================
% LOCAL FUNCTION
%% ============================================================

function summaryTable = createDistributionSummary(T, variableNames)

    numberOfVariables = numel(variableNames);

    sampleSize = repmat(height(T), numberOfVariables, 1);

    minimumValue = zeros(numberOfVariables, 1);
    maximumValue = zeros(numberOfVariables, 1);
    meanValue    = zeros(numberOfVariables, 1);
    medianValue  = zeros(numberOfVariables, 1);
    sdValue      = zeros(numberOfVariables, 1);
    covPercent   = zeros(numberOfVariables, 1);

    for i = 1:numberOfVariables

        values = T.(variableNames{i});

        minimumValue(i) = min(values, [], 'omitnan');
        maximumValue(i) = max(values, [], 'omitnan');
        meanValue(i)    = mean(values, 'omitnan');
        medianValue(i)  = median(values, 'omitnan');
        sdValue(i)      = std(values, 0, 'omitnan');

        if meanValue(i) ~= 0
            covPercent(i) = ...
                (sdValue(i) / abs(meanValue(i))) * 100;
        else
            covPercent(i) = NaN;
        end
    end

    summaryTable = table( ...
        string(variableNames(:)), ...
        sampleSize, ...
        round(minimumValue, 3), ...
        round(maximumValue, 3), ...
        round(meanValue, 3), ...
        round(medianValue, 3), ...
        round(sdValue, 3), ...
        round(covPercent, 3), ...
        'VariableNames', { ...
        'Variable', ...
        'N', ...
        'Min', ...
        'Max', ...
        'Mean', ...
        'Median', ...
        'SD', ...
        'COV_percent'});

end