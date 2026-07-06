%% ============================================================
% Random Shuffle + Split Dataset + Distribution Summary
%
% Export:
%   1) Train dataset + TRAIN_DISTRIBUTION sheet
%   2) Test dataset  + TEST_DISTRIBUTION sheet
%
% Selected Variables:
% D, t, fy, fc, L/D, N
%
% Statistics:
% N, Min, Max, Mean, Median, SD, COV(%)
% ============================================================

clear; clc; close all;

%% ===================== CONFIG =====================
input_file = "DATASET_N_S.xlsx";

train_file = "N_S_Training_DATASET_75.xlsx";
test_file  = "N_S_Testing_DATASET_25.xlsx";

random_state = 32;
train_ratio  = 0.75;

%% ===================== LOAD DATA =====================
df = readtable(input_file);

fprintf('\n====================================================\n');
fprintf('Original Data Shape\n');
fprintf('Rows    : %d\n', height(df));
fprintf('Columns : %d\n', width(df));
fprintf('====================================================\n');

%% ===================== KEEP ONLY REQUIRED COLUMNS =====================
requiredVars = {'D','t','Fy','Fc','L_D','N'};

% ถ้าในไฟล์ใช้ชื่อ L/D ให้ rename เป็น L_D
if any(strcmp(df.Properties.VariableNames,'L/D'))
    df.Properties.VariableNames{strcmp(df.Properties.VariableNames,'L/D')} = 'L_D';
end

df = df(:, requiredVars);

%% ===================== SHUFFLE DATA =====================
rng(random_state);

idx = randperm(height(df));
df_shuffled = df(idx, :);

%% ===================== SPLIT DATA =====================
n_train = floor(train_ratio * height(df_shuffled));

df_train = df_shuffled(1:n_train, :);
df_test  = df_shuffled(n_train+1:end, :);

fprintf('\nTraining Shape : %d rows x %d columns\n', ...
    height(df_train), width(df_train));

fprintf('Testing Shape  : %d rows x %d columns\n', ...
    height(df_test), width(df_test));

%% ===================== SUMMARY FUNCTION =====================
createSummary = @(T) table( ...
    requiredVars', ...
    repmat(height(T), length(requiredVars), 1), ...
    round(varfun(@min,    T, 'OutputFormat','uniform')',3), ...
    round(varfun(@max,    T, 'OutputFormat','uniform')',3), ...
    round(varfun(@mean,   T, 'OutputFormat','uniform')',3), ...
    round(varfun(@median, T, 'OutputFormat','uniform')',3), ...
    round(varfun(@std,    T, 'OutputFormat','uniform')',3), ...
    round( ...
        (varfun(@std, T, 'OutputFormat','uniform')' ./ ...
         varfun(@mean,T, 'OutputFormat','uniform')') * 100 ...
    ,3), ...
    'VariableNames', ...
    {'Variable','N','Min','Max','Mean','Median','SD','COV_percent'} ...
    );

%% ===================== CREATE SUMMARY =====================
summary_train = createSummary(df_train);
summary_test  = createSummary(df_test);

%% ===================== EXPORT TRAIN FILE =====================
writetable(df_train, train_file, ...
    'Sheet', 'TRAIN_DATA');

writetable(summary_train, train_file, ...
    'Sheet', 'TRAIN_DISTRIBUTION');

%% ===================== EXPORT TEST FILE =====================
writetable(df_test, test_file, ...
    'Sheet', 'TEST_DATA');

writetable(summary_test, test_file, ...
    'Sheet', 'TEST_DISTRIBUTION');

%% ===================== PRINT SUMMARY =====================
fprintf('\n====================================================\n');
fprintf('TRAIN DISTRIBUTION SUMMARY\n');
fprintf('====================================================\n');

disp(summary_train)

fprintf('\n====================================================\n');
fprintf('TEST DISTRIBUTION SUMMARY\n');
fprintf('====================================================\n');

disp(summary_test)

%% ===================== EXPORT MESSAGE =====================
fprintf('\n====================================================\n');
fprintf('EXPORT COMPLETED SUCCESSFULLY!\n');
fprintf('====================================================\n');

fprintf('\nSaved Files:\n');
fprintf('1. %s\n', train_file);
fprintf('   - TRAIN_DATA\n');
fprintf('   - TRAIN_DISTRIBUTION\n\n');

fprintf('2. %s\n', test_file);
fprintf('   - TEST_DATA\n');
fprintf('   - TEST_DISTRIBUTION\n');