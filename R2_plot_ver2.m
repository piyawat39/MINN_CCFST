clc;
clear;
close all;

%% ===================== Select Excel file =====================
% The supplied workbook is selected automatically when it is in the same
% folder as this script. Otherwise, MATLAB asks the user to select a file.
scriptDir = fileparts(mfilename('fullpath'));
defaultFile = fullfile(scriptDir, 'Model accuracy ver 2.xlsx');

if isfile(defaultFile)
    fullFile = defaultFile;
else
    [fileName, filePath] = uigetfile('*.xlsx', 'Select model-accuracy Excel file');
    if isequal(fileName, 0)
        error('No Excel file selected.');
    end
    fullFile = fullfile(filePath, fileName);
end

%% ===================== Read Excel sheets =====================
trainData = readtable(fullFile, ...
    'Sheet', 'Training', ...
    'VariableNamingRule', 'preserve');

testData = readtable(fullFile, ...
    'Sheet', 'Testing', ...
    'VariableNamingRule', 'preserve');

%% ===================== Locate required columns =====================
trainCols = string(trainData.Properties.VariableNames);
testCols  = string(testData.Properties.VariableNames);

actualTrainIdx = find(strcmpi(strtrim(trainCols), 'N'), 1);
actualTestIdx  = find(strcmpi(strtrim(testCols),  'N'), 1);

if isempty(actualTrainIdx) || isempty(actualTestIdx)
    error('Column "N" was not found in both Training and Testing sheets.');
end

% In this workbook, all columns after N are model predictions (KNN--AISC).
trainModelIdx = (actualTrainIdx + 1):width(trainData);
testModelIdx  = (actualTestIdx  + 1):width(testData);

if isempty(trainModelIdx) || isempty(testModelIdx)
    error('No model-prediction columns were found after column "N".');
end

trainModels = trainCols(trainModelIdx);
testModels  = testCols(testModelIdx);

% Match model columns without case sensitivity (e.g., XGBoost/Xgboost).
commonModels = strings(0, 1);
matchedTrainIdx = [];
matchedTestIdx  = [];

for i = 1:numel(trainModels)
    j = find(strcmpi(strtrim(testModels), strtrim(trainModels(i))), 1);
    if ~isempty(j)
        commonModels(end+1, 1) = trainModels(i); %#ok<SAGROW>
        matchedTrainIdx(end+1) = trainModelIdx(i); %#ok<SAGROW>
        matchedTestIdx(end+1)  = testModelIdx(j); %#ok<SAGROW>
    else
        warning('Model "%s" exists in Training but not Testing; it will be skipped.', trainModels(i));
    end
end

if isempty(commonModels)
    error('No matching model columns were found in Training and Testing sheets.');
end

%% ===================== Actual values =====================
yTrain = double(trainData{:, actualTrainIdx});
yTest  = double(testData{:, actualTestIdx});

%% ===================== Output folder =====================
[excelDir, ~, ~] = fileparts(fullFile);
outDir = fullfile(excelDir, 'Accuracy_indicator');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% ===================== Zoom range =====================
zoomRange = [-200 5000];

%% ===================== Loop each algorithm =====================
for a = 1:numel(commonModels)

    alg = char(commonModels(a));

    %% ===================== Prediction =====================
    yPredTrain = double(trainData{:, matchedTrainIdx(a)});
    yPredTest  = double(testData{:, matchedTestIdx(a)});

    %% ===================== Metrics =====================
    metTrain = calMetrics(yTrain, yPredTrain);
    metTest  = calMetrics(yTest,  yPredTest);

    %% ===================== Number of data =====================
    nTrain = sum(isfinite(yTrain) & isfinite(yPredTrain));
    nTest  = sum(isfinite(yTest)  & isfinite(yPredTest));

    %% ===================== Axis limit =====================
    allVal = [yTrain; yTest; yPredTrain; yPredTest];
    maxVal = max(allVal, [], 'omitnan');
    limMax = max(1000, ceil(maxVal/1000)*1000);

    xLineMain = linspace(-200, limMax, 500);
    xLineZoom = linspace(zoomRange(1), zoomRange(2), 200);

    %% ===================== Main figure =====================
    fig = figure('Color', 'w', 'Position', [100 100 750 680]);
    mainAx = axes(fig);
    hold(mainAx, 'on');

    scatter(mainAx, yTrain, yPredTrain, 40, 's', ...
        'LineWidth', 1.2, 'MarkerEdgeColor', [0 0.4470 0.7410]);
    scatter(mainAx, yTest, yPredTest, 45, '^', ...
        'LineWidth', 1.2, 'MarkerEdgeColor', [1 0 0]);

    plot(mainAx, xLineMain, xLineMain, 'k-', 'LineWidth', 1.5);
    plot(mainAx, xLineMain, 1.2.*xLineMain, 'k--', 'LineWidth', 1.2);
    plot(mainAx, xLineMain, 0.8.*xLineMain, 'k--', 'LineWidth', 1.2);

    rectangle(mainAx, 'Position', [zoomRange(1), zoomRange(1), ...
        diff(zoomRange), diff(zoomRange)], 'EdgeColor', 'k', ...
        'LineStyle', '--', 'LineWidth', 1.2);

    %% ===================== Metrics text =====================
    txt = sprintf([ ...
        'Train (n = %d)\n' ...
        'R^2 = %.2f%%\nRMSE = %.2f\nMSE = %.2f\nMAE = %.2f\nA20 = %.2f%%\n\n' ...
        'Test (n = %d)\n' ...
        'R^2 = %.2f%%\nRMSE = %.2f\nMSE = %.2f\nMAE = %.2f\nA20 = %.2f%%'], ...
        nTrain, metTrain.R2*100, metTrain.RMSE, metTrain.MSE, ...
        metTrain.MAE, metTrain.A20*100, nTest, metTest.R2*100, ...
        metTest.RMSE, metTest.MSE, metTest.MAE, metTest.A20*100);

    text(mainAx, 0.05*limMax, 0.95*limMax, txt, ...
        'FontName', 'Times New Roman', 'FontSize', 14, ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', ...
        'EdgeColor', [0.5 0.5 0.5]);

    xlabel(mainAx, 'Nu (Real value, kN)', ...
        'FontName', 'Times New Roman', 'FontSize', 22);
    ylabel(mainAx, 'Nu (Predicted value, kN)', ...
        'FontName', 'Times New Roman', 'FontSize', 22);
    legend(mainAx, {'Train','Test','y = x','+20%','-20%'}, ...
        'Location', 'northeast', 'FontName', 'Times New Roman', ...
        'FontSize', 14);

    xlim(mainAx, [-200 limMax]);
    ylim(mainAx, [-200 limMax]);
    axis(mainAx, 'square');
    grid(mainAx, 'off');
    box(mainAx, 'on');
    set(mainAx, 'FontName', 'Times New Roman', ...
        'FontSize', 12, 'LineWidth', 1.2);

    % Create the algorithm title after the axes formatting so its font size
    % is not reset by the axes FontSize setting above.
    hTitle = title(mainAx, alg, 'FontName', 'Times New Roman', ...
        'FontSize', 20, 'FontWeight', 'bold', 'Interpreter', 'none');
    hTitle.FontSizeMode = 'manual';

    %% ===================== ZOOM INSET =====================
    insetAx = axes('Position', [0.56 0.18 0.34 0.34]);
    hold(insetAx, 'on');

    scatter(insetAx, yTrain, yPredTrain, 20, 's', ...
        'LineWidth', 1.0, 'MarkerEdgeColor', [0 0.4470 0.7410]);
    scatter(insetAx, yTest, yPredTest, 24, '^', ...
        'LineWidth', 1.0, 'MarkerEdgeColor', [1 0 0]);
    plot(insetAx, xLineZoom, xLineZoom, 'k-', 'LineWidth', 1.0);
    plot(insetAx, xLineZoom, 1.2.*xLineZoom, 'k--', 'LineWidth', 0.8);
    plot(insetAx, xLineZoom, 0.8.*xLineZoom, 'k--', 'LineWidth', 0.8);

    xlim(insetAx, zoomRange);
    ylim(insetAx, zoomRange);
    axis(insetAx, 'square');
    grid(insetAx, 'off');
    box(insetAx, 'on');
    set(insetAx, 'FontName', 'Times New Roman', ...
        'FontSize', 10, 'LineWidth', 1);
    title(insetAx, 'Zoom: -200 to 5000 kN', ...
        'FontName', 'Times New Roman', 'FontSize', 12, ...
        'FontWeight', 'bold');

    %% ===================== Export =====================
    safeName = regexprep(alg, '[^a-zA-Z0-9_]', '_');
    exportgraphics(fig, fullfile(outDir, [safeName '_Parity_Hollow.png']), ...
        'Resolution', 1200);
    exportgraphics(fig, fullfile(outDir, [safeName '_Parity_Hollow.tiff']), ...
        'Resolution', 1200);
    close(fig);
end

fprintf('Finished: %d model plots exported to:\n%s\n', ...
    numel(commonModels), outDir);

%% =====================================================
%% ===================== FUNCTION ======================
%% =====================================================
function met = calMetrics(yTrue, yPred)
    valid = isfinite(yTrue) & isfinite(yPred);
    yTrue = yTrue(valid);
    yPred = yPred(valid);

    if isempty(yTrue)
        met = struct('MAE', NaN, 'MSE', NaN, 'RMSE', NaN, ...
            'R2', NaN, 'A20', NaN);
        return;
    end

    err = yPred - yTrue;
    met.MAE = mean(abs(err));
    met.MSE = mean(err.^2);
    met.RMSE = sqrt(met.MSE);

    SSres = sum(err.^2);
    SStot = sum((yTrue - mean(yTrue)).^2);
    if SStot > 0
        met.R2 = 1 - SSres/SStot;
    else
        met.R2 = NaN;
    end

    nonzero = yTrue ~= 0;
    ratio = yPred(nonzero) ./ yTrue(nonzero);
    met.A20 = mean(ratio >= 0.8 & ratio <= 1.2);
end
