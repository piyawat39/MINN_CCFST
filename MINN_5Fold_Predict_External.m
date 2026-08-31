clear; clc; close all;

%% ============================================================
% ANN AND MINN 5-FOLD ENSEMBLES: EXTERNAL DATA PREDICTION
%
% Required model structure:
% Results_MINN_N_S_5Fold/
%   Fold_01/MINN_Model_Fold_01.mat
%   ...
%   Fold_05/MINN_Model_Fold_05.mat
%
% External Excel input (sheet TEST_DATA):
% D, t, fy, fc, L/D, N
%% ============================================================

%% 1) SETTINGS
externalFile = 'N_S_Testing_DATASET_External.xlsx';
externalSheet = 'TEST_DATA';
modelRoot = 'Results_MINN_N_S_5Fold';
annModelRoot = 'Results_ANN_N_S_5Fold';
K = 5;

outputDir = 'Results_ANN_MINN_External_Prediction';
outputExcel = fullfile(outputDir,'ANN_MINN_5Fold_External_Predictions.xlsx');
outputFigure = fullfile(outputDir,'ANN_MINN_5Fold_External_Experimental_vs_Predicted.png');

if ~exist(outputDir,'dir'), mkdir(outputDir); end

%% 2) LOCATE INPUT FILE
scriptFolder = fileparts(mfilename('fullpath'));
externalFile = locateFile(externalFile,{pwd,scriptFolder});

%% 3) LOCATE MODEL ROOT
candidateRoots = {modelRoot, ...
    fullfile(pwd,modelRoot), ...
    fullfile(scriptFolder,modelRoot)};

resolvedModelRoot = '';
for i = 1:numel(candidateRoots)
    if hasFiveFoldFiles(candidateRoots{i},K,'MINN')
        resolvedModelRoot = candidateRoots{i};
        break;
    end
end

if isempty(resolvedModelRoot)
    selectedFolder = uigetdir(scriptFolder, ...
        'Select Results_MINN_N_S_5Fold containing Fold_01 to Fold_05');
    if isequal(selectedFolder,0)
        error('The five-fold model folder was not selected.');
    end
    if ~hasFiveFoldFiles(selectedFolder,K,'MINN')
        error('The selected folder does not contain all five fold models.');
    end
    resolvedModelRoot = selectedFolder;
end

fprintf('External data : %s\n',externalFile);
fprintf('Model folder  : %s\n',resolvedModelRoot);

%% 4) LOCATE ANN FIVE-FOLD MODEL FOLDER
annCandidateRoots = {annModelRoot,fullfile(pwd,annModelRoot), ...
    fullfile(scriptFolder,annModelRoot)};
resolvedANNRoot = '';
for i = 1:numel(annCandidateRoots)
    if hasFiveFoldFiles(annCandidateRoots{i},K,'ANN')
        resolvedANNRoot = annCandidateRoots{i};
        break;
    end
end
if isempty(resolvedANNRoot)
    selectedFolder = uigetdir(scriptFolder, ...
        'Select Results_ANN_N_S_5Fold containing Fold_01 to Fold_05');
    if isequal(selectedFolder,0)
        error('The ANN five-fold model folder was not selected.');
    end
    if ~hasFiveFoldFiles(selectedFolder,K,'ANN')
        error('The selected folder does not contain all five ANN fold models.');
    end
    resolvedANNRoot = selectedFolder;
end
fprintf('ANN folder   : %s\n',resolvedANNRoot);

%% 5) READ EXTERNAL DATA
T = readtable(externalFile,'Sheet',externalSheet, ...
    'VariableNamingRule','preserve');
T.Properties.VariableNames = matlab.lang.makeValidName( ...
    strtrim(T.Properties.VariableNames));

requiredVars = {'D','t','fy','fc','L_D','N'};
for i = 1:numel(requiredVars)
    if ~ismember(requiredVars{i},T.Properties.VariableNames)
        error('Required variable "%s" was not found in sheet %s.', ...
            requiredVars{i},externalSheet);
    end
end

X = [T.D T.t T.fy T.fc T.L_D];
yExp = T.N;

validRows = all(isfinite(X),2) & isfinite(yExp);
if ~all(validRows)
    warning('%d invalid row(s) containing NaN/Inf were removed.',sum(~validRows));
    T = T(validRows,:);
    X = X(validRows,:);
    yExp = yExp(validRows,:);
end

nExternal = size(X,1);
if nExternal == 0
    error('No valid external observations were found.');
end

%% 6) LOAD EACH MINN FOLD AND PREDICT
minnPredictionByFold = nan(nExternal,K);
minnModelFiles = strings(K,1);

for fold = 1:K
    foldFolder = fullfile(resolvedModelRoot,sprintf('Fold_%02d',fold));
    pattern = sprintf('MINN_Model_Fold_%02d*.mat',fold);
    matches = dir(fullfile(foldFolder,pattern));

    if isempty(matches)
        error('Model for Fold %d was not found in %s.',fold,foldFolder);
    end
    if numel(matches) > 1
        [~,idxNewest] = max([matches.datenum]);
        matches = matches(idxNewest);
        warning('Multiple files found for Fold %d; using newest: %s', ...
            fold,matches.name);
    end

    modelFile = fullfile(matches.folder,matches.name);
    minnModelFiles(fold) = string(modelFile);
    S = load(modelFile,'net','muX','stdX','muy','stdy');

    requiredModelVars = {'net','muX','stdX','muy','stdy'};
    for i = 1:numel(requiredModelVars)
        if ~isfield(S,requiredModelVars{i})
            error('Variable "%s" is missing from %s.', ...
                requiredModelVars{i},modelFile);
        end
    end

    if numel(S.muX) ~= size(X,2) || numel(S.stdX) ~= size(X,2)
        error('Fold %d normalization expects %d inputs, but external data has %d.', ...
            fold,numel(S.muX),size(X,2));
    end

    Xn = (X-S.muX(:)')./S.stdX(:)';
    xdl = dlarray(Xn','CB');
    yNorm = forward(S.net,xdl,'Outputs','output');
    minnPredictionByFold(:,fold) = ...
        gather(extractdata(yNorm))'.*S.stdy + S.muy;

    fprintf('Fold %d prediction completed: %s\n',fold,matches.name);
end

%% 7) PREDICT WITH ANN FIVE-FOLD ENSEMBLE
annPredictionByFold = nan(nExternal,K);
annModelFiles = strings(K,1);
for fold = 1:K
    annFoldFolder = fullfile(resolvedANNRoot,sprintf('Fold_%02d',fold));
    annPattern = sprintf('ANN_Model_Fold_%02d*.mat',fold);
    annMatches = dir(fullfile(annFoldFolder,annPattern));
    if isempty(annMatches)
        error('ANN model for Fold %d was not found in %s.',fold,annFoldFolder);
    end
    if numel(annMatches)>1
        [~,idxNewest] = max([annMatches.datenum]);
        annMatches = annMatches(idxNewest);
        warning('Multiple ANN files found for Fold %d; using newest: %s', ...
            fold,annMatches.name);
    end
    annFile = fullfile(annMatches.folder,annMatches.name);
    annModelFiles(fold) = string(annFile);
    A = load(annFile,'net','muX','stdX','muy','stdy');
    requiredANNVars = {'net','muX','stdX','muy','stdy'};
    for i = 1:numel(requiredANNVars)
        if ~isfield(A,requiredANNVars{i})
            error('Required variable "%s" is missing from %s.', ...
                requiredANNVars{i},annFile);
        end
    end
    annMuX = A.muX(:)'; annStdX = A.stdX(:)';
    if numel(annMuX)~=size(X,2) || numel(annStdX)~=size(X,2)
        error('ANN Fold %d expects %d inputs, but external data has %d.', ...
            fold,numel(annMuX),size(X,2));
    end
    XnANN = (X-annMuX)./annStdX;
    xdlANN = dlarray(XnANN','CB');
    annYNorm = forward(A.net,xdlANN,'Outputs','output');
    annPredictionByFold(:,fold) = ...
        gather(extractdata(annYNorm))'.*A.stdy + A.muy;
    fprintf('ANN Fold %d prediction completed: %s\n',fold,annMatches.name);
end

%% 8) ENSEMBLE PREDICTIONS AND ACCURACY METRICS
minnPred = mean(minnPredictionByFold,2,'omitnan');
annPred = mean(annPredictionByFold,2,'omitnan');

minnResidual = minnPred-yExp;
annResidual = annPred-yExp;
minnRatio = minnPred./yExp;
annRatio = annPred./yExp;

MINN_Metrics = calculateMetrics(yExp,minnPred);
ANN_Metrics = calculateMetrics(yExp,annPred);

fprintf('\n================ EXTERNAL VALIDATION ================\n');
fprintf('N          = %d\n',nExternal);
fprintf('ANN  | R2 = %.6f | RMSE = %.3f kN | A20 = %.3f %%\n', ...
    ANN_Metrics.R2,ANN_Metrics.RMSE,ANN_Metrics.A20_percent);
fprintf('MINN | R2 = %.6f | RMSE = %.3f kN | A20 = %.3f %%\n', ...
    MINN_Metrics.R2,MINN_Metrics.RMSE,MINN_Metrics.A20_percent);
fprintf('=====================================================\n');

%% 9) TRAINING-RANGE CHECK
% Ranges used by the associated GUI/training database.
lowerBound = [47.28 0.52 185.70 5.51 0.81];
upperBound = [1020.00 16.53 853.00 179.55 7.46];
outsideByVariable = X<lowerBound | X>upperBound;
outsideTrainingRange = any(outsideByVariable,2);

if any(outsideTrainingRange)
    fprintf('Warning: %d/%d external rows are outside at least one training range.\n', ...
        sum(outsideTrainingRange),nExternal);
end

%% 10) EXPORT EXCEL
PredictionTable = T;
for fold = 1:K
    PredictionTable.(sprintf('ANN_pred_Fold_%02d',fold)) = ...
        annPredictionByFold(:,fold);
    PredictionTable.(sprintf('MINN_pred_Fold_%02d',fold)) = ...
        minnPredictionByFold(:,fold);
end
PredictionTable.ANN_pred_5Fold_Mean = annPred;
PredictionTable.MINN_pred_5Fold_Mean = minnPred;
PredictionTable.ANN_Residual_kN = annResidual;
PredictionTable.MINN_Residual_kN = minnResidual;
PredictionTable.ANN_Absolute_Error_kN = abs(annResidual);
PredictionTable.MINN_Absolute_Error_kN = abs(minnResidual);
PredictionTable.ANN_pred_over_N_exp = annRatio;
PredictionTable.MINN_pred_over_N_exp = minnRatio;
PredictionTable.ANN_Within_A20 = annRatio>=0.80 & annRatio<=1.20;
PredictionTable.MINN_Within_A20 = minnRatio>=0.80 & minnRatio<=1.20;
PredictionTable.Outside_Training_Range = outsideTrainingRange;
PredictionTable.Outside_D = outsideByVariable(:,1);
PredictionTable.Outside_t = outsideByVariable(:,2);
PredictionTable.Outside_fy = outsideByVariable(:,3);
PredictionTable.Outside_fc = outsideByVariable(:,4);
PredictionTable.Outside_L_D = outsideByVariable(:,5);

ANN_Row = struct2table(ANN_Metrics,'AsArray',true);
MINN_Row = struct2table(MINN_Metrics,'AsArray',true);
ANN_Row.Model = "ANN"; MINN_Row.Model = "MINN";
MetricsTable = [ANN_Row;MINN_Row];
MetricsTable = movevars(MetricsTable,'Model','Before',1);
ModelFilesTable = table((1:K)',annModelFiles, ...
    minnModelFiles,'VariableNames',{'Fold','ANN_Model_File','MINN_Model_File'});

writetable(PredictionTable,outputExcel,'Sheet','EXTERNAL_PREDICTIONS');
writetable(MetricsTable,outputExcel,'Sheet','EXTERNAL_METRICS');
writetable(ModelFilesTable,outputExcel,'Sheet','MODEL_FILES');

%% 11) EXPERIMENTAL VS PREDICTED FIGURE
f = figure('Color','w','Position',[150 120 820 700]);
scatter(yExp,annPred,60,'o','filled','MarkerFaceAlpha',0.75); hold on;
scatter(yExp,minnPred,65,'^','filled','MarkerFaceAlpha',0.75);
plotLimit = [min([yExp;annPred;minnPred]) max([yExp;annPred;minnPred])];
plot(plotLimit,plotLimit,'k--','LineWidth',1.5);
hold off; grid on; box on; axis equal;
xlim(plotLimit); ylim(plotLimit);
xlabel('Experimental N_u (kN)','FontSize',13);
ylabel('Predicted N_u (kN)','FontSize',13);
legend('ANN','MINN','1:1 line','Location','best');
title({'ANN and MINN Five-Fold Ensembles: External Validation', ...
    sprintf('ANN RMSE = %.2f kN | MINN RMSE = %.2f kN', ...
    ANN_Metrics.RMSE,MINN_Metrics.RMSE)},'FontSize',14);
exportgraphics(f,outputFigure,'Resolution',1200);

fprintf('\nSaved Excel : %s\n',outputExcel);
fprintf('Saved figure: %s\n',outputFigure);

%% ============================================================
% LOCAL FUNCTIONS
%% ============================================================
function resolved = locateFile(fileName,searchFolders)
resolved = '';
if isfile(fileName)
    resolved = fileName;
    return;
end
for i = 1:numel(searchFolders)
    candidate = fullfile(searchFolders{i},fileName);
    if isfile(candidate)
        resolved = candidate;
        return;
    end
end
if isempty(resolved)
    [selectedFile,selectedFolder] = uigetfile({'*.xlsx','Excel files'}, ...
        sprintf('Select %s',fileName));
    if isequal(selectedFile,0)
        error('External Excel file was not selected.');
    end
    resolved = fullfile(selectedFolder,selectedFile);
end
end

function tf = hasFiveFoldFiles(rootFolder,K,modelPrefix)
tf = isfolder(rootFolder);
if ~tf, return; end
for fold = 1:K
    foldFolder = fullfile(rootFolder,sprintf('Fold_%02d',fold));
    pattern = sprintf('%s_Model_Fold_%02d*.mat',modelPrefix,fold);
    if ~isfolder(foldFolder) || isempty(dir(fullfile(foldFolder,pattern)))
        tf = false;
        return;
    end
end
end

function M = calculateMetrics(observed,predicted)
observed = observed(:); predicted = predicted(:);
valid = isfinite(observed) & isfinite(predicted);
observed = observed(valid); predicted = predicted(valid);

errorValue = predicted-observed;
M.N = numel(observed);
M.MAE = mean(abs(errorValue));
M.MSE = mean(errorValue.^2);
M.RMSE = sqrt(M.MSE);

denominator = sum((observed-mean(observed)).^2);
if denominator>0
    M.R2 = 1-sum(errorValue.^2)/denominator;
else
    M.R2 = NaN;
end

nonzero = observed~=0;
if any(nonzero)
    M.MAPE_percent = mean(abs(errorValue(nonzero)./observed(nonzero)))*100;
else
    M.MAPE_percent = NaN;
end

predictionRatio = predicted./observed;
M.A20_percent = mean(predictionRatio>=0.80 & predictionRatio<=1.20)*100;
M.MeanRatio = mean(predictionRatio,'omitnan');
M.SDRatio = std(predictionRatio,'omitnan');
end
