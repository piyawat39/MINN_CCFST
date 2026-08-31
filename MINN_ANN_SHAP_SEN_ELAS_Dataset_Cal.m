%% ============================================================
% Compare Sensitivity, Elasticity and SHAP between ANN and MINN
% Separate TRAIN and TEST data
% Sensitivity/Elasticity using Forward +1% input increase
% Export results to Excel
%% ============================================================

clear; clc; close all;

% Load fold models exported by ANN_N_S_5_Fold.m and MINN_N_S_5_Fold.m.
% Neither model is retrained here.
annModelRoot = 'Results_ANN_N_S_5Fold';
minnModelRoot = 'Results_MINN_N_S_5Fold';

trainFile = 'N_S_Training_DATASET_75.xlsx';
testFile  = 'N_S_Testing_DATASET_25.xlsx';
outDir = 'SHAP_SEN_ELAS_Dataset';
outExcel = fullfile(outDir, ...
    'ANN_MINN_Sensitivity_Elasticity_SHAP_Train_Test_ver2.xlsx');

outDir = 'SHAP_SEN_ELAS_Dataset';

if ~exist(outDir,'dir')
    mkdir(outDir);
end

fontName = 'Times New Roman';
fontSize = 12;

doExportFigure = true;
outDPI = 1200;

% ใช้ Forward +1% เหมือน Python Tree model
relStep_default = 0.01;

% SHAP permutation
nPermSHAP = 100;   % ถ้าช้า ลดเป็น 50 ได้

rng(500);

%% ---------------- LOAD DATA ----------------
[Ttr,trainSheetUsed] = readDataSheetAuto(trainFile);
[Tte,testSheetUsed] = readDataSheetAuto(testFile);
fprintf('Training-data sheet selected = %s\n',trainSheetUsed);
fprintf('Testing-data sheet selected  = %s\n',testSheetUsed);

%% ---------------- LOAD MODELS ----------------
scriptFolder = fileparts(mfilename('fullpath'));
annModelRoot = locateFiveFoldRoot(annModelRoot,'ANN',scriptFolder);
minnModelRoot = locateFiveFoldRoot(minnModelRoot,'MINN',scriptFolder);

K=5;
ANN.bestFoldModels=cell(K,1); ANN.bestFoldMuX=cell(K,1);
ANN.bestFoldStdX=cell(K,1); ANN.bestFoldMuy=nan(K,1);
ANN.bestFoldStdy=nan(K,1);
MINN.bestModels=cell(K,1); MINN.bestNorm=cell(K,1);

for fold=1:K
    annFile=selectFoldModelFile(annModelRoot,'ANN',fold);
    A=load(annFile,'net','muX','stdX','muy','stdy');
    checkModelFields(A,sprintf('ANN Fold %d',fold));
    ANN.bestFoldModels{fold}=A.net;
    ANN.bestFoldMuX{fold}=A.muX;
    ANN.bestFoldStdX{fold}=A.stdX;
    ANN.bestFoldMuy(fold)=A.muy;
    ANN.bestFoldStdy(fold)=A.stdy;

    minnFile=selectFoldModelFile(minnModelRoot,'MINN',fold);
    M=load(minnFile,'net','muX','stdX','muy','stdy');
    checkModelFields(M,sprintf('MINN Fold %d',fold));
    MINN.bestModels{fold}=M.net;
    MINN.bestNorm{fold}=struct('muX',M.muX,'stdX',M.stdX, ...
        'muy',M.muy,'stdy',M.stdy);
end

checkANNEnsembleFields(ANN);
checkMINNEnsembleFields(MINN);

nFeat_ANN  = numel(ANN.bestFoldMuX{1});
nFeat_MINN = numel(MINN.bestNorm{1}.muX);

if nFeat_ANN ~= nFeat_MINN
    error('ANN and MINN have different numbers of input features.');
end

nFeat = nFeat_ANN;

if ~(nFeat == 5 || nFeat == 6)
    error('This script supports only 5-input or 6-input models.');
end

fprintf('Number of input features = %d\n', nFeat);
fprintf('Loaded trained ANN ensemble  = %d fold models\n',numel(ANN.bestFoldModels));
fprintf('Loaded trained MINN ensemble = %d fold models\n',numel(MINN.bestModels));
fprintf('ANN model folder  : %s\n',annModelRoot);
fprintf('MINN model folder : %s\n',minnModelRoot);

%% ---------------- BUILD TRAIN INPUT ----------------
[tblX_train, featShow] = buildFeatureTable_auto(Ttr, nFeat);
tblX_train = rmmissing(tblX_train);
Xtrain = table2array(tblX_train);

%% ---------------- BUILD TEST INPUT ----------------
[tblX_test, ~] = buildFeatureTable_auto(Tte, nFeat);
tblX_test = rmmissing(tblX_test);
Xtest = table2array(tblX_test);

fprintf('Train rows used = %d\n', size(Xtrain,1));
fprintf('Test rows used  = %d\n', size(Xtest,1));

disp('Feature order used:');
disp(tblX_train.Properties.VariableNames);

%% ============================================================
% PREDICTION
%% ============================================================

fprintf('\nPredicting ANN and MINN outputs...\n');

Y_ANN_train = predictANNEnsemble_vec(Xtrain,ANN);

Y_MINN_train = predictMINNEnsemble_vec(Xtrain,MINN.bestModels,MINN.bestNorm);

Y_ANN_test = predictANNEnsemble_vec(Xtest,ANN);

Y_MINN_test = predictMINNEnsemble_vec(Xtest,MINN.bestModels,MINN.bestNorm);

%% ============================================================
% SENSITIVITY + ELASTICITY : FORWARD +1%
%% ============================================================

fprintf('\nCalculating ANN Train sensitivity and elasticity...\n');

[E_ANN_train,dY_ANN_train,pct_ANN_train] = ...
    calculateElasticityForward_ANNEnsemble(Xtrain,ANN,relStep_default);

fprintf('\nCalculating ANN Test sensitivity and elasticity...\n');

[E_ANN_test,dY_ANN_test,pct_ANN_test] = ...
    calculateElasticityForward_ANNEnsemble(Xtest,ANN,relStep_default);

fprintf('\nCalculating MINN Train sensitivity and elasticity...\n');

[E_MINN_train,dY_MINN_train,pct_MINN_train] = ...
    calculateElasticityForward_MINNEnsemble( ...
    Xtrain,MINN.bestModels,MINN.bestNorm,relStep_default);

fprintf('\nCalculating MINN Test sensitivity and elasticity...\n');

[E_MINN_test,dY_MINN_test,pct_MINN_test] = ...
    calculateElasticityForward_MINNEnsemble( ...
    Xtest,MINN.bestModels,MINN.bestNorm,relStep_default);

%% ============================================================
% SHAP
%% ============================================================

fprintf('\nCalculating ANN SHAP for Train data...\n');

SHAP_ANN_train = calculatePermutationSHAP_ANNEnsemble( ...
    Xtrain,Xtrain,ANN,nPermSHAP);

fprintf('\nCalculating ANN SHAP for Test data...\n');

SHAP_ANN_test = calculatePermutationSHAP_ANNEnsemble( ...
    Xtest,Xtrain,ANN,nPermSHAP);

fprintf('\nCalculating MINN SHAP for Train data...\n');

SHAP_MINN_train = calculatePermutationSHAP_MINNEnsemble( ...
    Xtrain,Xtrain,MINN.bestModels,MINN.bestNorm,nPermSHAP);

fprintf('\nCalculating MINN SHAP for Test data...\n');

SHAP_MINN_test = calculatePermutationSHAP_MINNEnsemble( ...
    Xtest,Xtrain,MINN.bestModels,MINN.bestNorm,nPermSHAP);

%% ============================================================
% PREDICTION TABLES
%% ============================================================

Tpred_ANN_train = tblX_train;
Tpred_ANN_train.Predicted_Nu = Y_ANN_train;

Tpred_ANN_test = tblX_test;
Tpred_ANN_test.Predicted_Nu = Y_ANN_test;

Tpred_MINN_train = tblX_train;
Tpred_MINN_train.Predicted_Nu = Y_MINN_train;

Tpred_MINN_test = tblX_test;
Tpred_MINN_test.Predicted_Nu = Y_MINN_test;

%% ============================================================
% SUMMARY TABLES
%% ============================================================

ANN_SenElas_Train = createSenElasSummary( ...
    featShow, dY_ANN_train, pct_ANN_train, E_ANN_train, ...
    'ANN', 'Train', relStep_default);

ANN_SenElas_Test = createSenElasSummary( ...
    featShow, dY_ANN_test, pct_ANN_test, E_ANN_test, ...
    'ANN', 'Test', relStep_default);

MINN_SenElas_Train = createSenElasSummary( ...
    featShow, dY_MINN_train, pct_MINN_train, E_MINN_train, ...
    'MINN', 'Train', relStep_default);

MINN_SenElas_Test = createSenElasSummary( ...
    featShow, dY_MINN_test, pct_MINN_test, E_MINN_test, ...
    'MINN', 'Test', relStep_default);

All_Sen_Elasticity = [
    ANN_SenElas_Train;
    ANN_SenElas_Test;
    MINN_SenElas_Train;
    MINN_SenElas_Test
];

%% ============================================================
% RAW TABLES
%% ============================================================

Sensitivity_ANN_Train = array2table(dY_ANN_train, ...
    'VariableNames', matlab.lang.makeValidName(featShow));
Sensitivity_ANN_Test = array2table(dY_ANN_test, ...
    'VariableNames', matlab.lang.makeValidName(featShow));

Sensitivity_MINN_Train = array2table(dY_MINN_train, ...
    'VariableNames', matlab.lang.makeValidName(featShow));
Sensitivity_MINN_Test = array2table(dY_MINN_test, ...
    'VariableNames', matlab.lang.makeValidName(featShow));

OutputChange_ANN_Train = array2table(pct_ANN_train, ...
    'VariableNames', matlab.lang.makeValidName(featShow));
OutputChange_ANN_Test = array2table(pct_ANN_test, ...
    'VariableNames', matlab.lang.makeValidName(featShow));

OutputChange_MINN_Train = array2table(pct_MINN_train, ...
    'VariableNames', matlab.lang.makeValidName(featShow));
OutputChange_MINN_Test = array2table(pct_MINN_test, ...
    'VariableNames', matlab.lang.makeValidName(featShow));

Elasticity_ANN_Train = array2table(E_ANN_train, ...
    'VariableNames', matlab.lang.makeValidName(featShow));
Elasticity_ANN_Test = array2table(E_ANN_test, ...
    'VariableNames', matlab.lang.makeValidName(featShow));

Elasticity_MINN_Train = array2table(E_MINN_train, ...
    'VariableNames', matlab.lang.makeValidName(featShow));
Elasticity_MINN_Test = array2table(E_MINN_test, ...
    'VariableNames', matlab.lang.makeValidName(featShow));

%% ============================================================
% SHAP TABLES
%% ============================================================

SHAP_ANN_Train_Table = array2table(SHAP_ANN_train, ...
    'VariableNames', matlab.lang.makeValidName(featShow));
SHAP_ANN_Test_Table = array2table(SHAP_ANN_test, ...
    'VariableNames', matlab.lang.makeValidName(featShow));

SHAP_MINN_Train_Table = array2table(SHAP_MINN_train, ...
    'VariableNames', matlab.lang.makeValidName(featShow));
SHAP_MINN_Test_Table = array2table(SHAP_MINN_test, ...
    'VariableNames', matlab.lang.makeValidName(featShow));

ANN_SHAP_Summary_Train = createSHAPSummary( ...
    featShow, SHAP_ANN_train, 'ANN', 'Train');

ANN_SHAP_Summary_Test = createSHAPSummary( ...
    featShow, SHAP_ANN_test, 'ANN', 'Test');

MINN_SHAP_Summary_Train = createSHAPSummary( ...
    featShow, SHAP_MINN_train, 'MINN', 'Train');

MINN_SHAP_Summary_Test = createSHAPSummary( ...
    featShow, SHAP_MINN_test, 'MINN', 'Test');

All_SHAP_Summary = [
    ANN_SHAP_Summary_Train;
    ANN_SHAP_Summary_Test;
    MINN_SHAP_Summary_Train;
    MINN_SHAP_Summary_Test
];

%% ============================================================
% ALL SUMMARY
%% ============================================================

All_Summary = table( ...
    featShow(:), ...
    mean(abs(dY_ANN_train),1,'omitnan')', ...
    mean(abs(dY_ANN_test),1,'omitnan')', ...
    mean(abs(dY_MINN_train),1,'omitnan')', ...
    mean(abs(dY_MINN_test),1,'omitnan')', ...
    mean(abs(pct_ANN_train),1,'omitnan')', ...
    mean(abs(pct_ANN_test),1,'omitnan')', ...
    mean(abs(pct_MINN_train),1,'omitnan')', ...
    mean(abs(pct_MINN_test),1,'omitnan')', ...
    mean(abs(E_ANN_train),1,'omitnan')', ...
    mean(abs(E_ANN_test),1,'omitnan')', ...
    mean(abs(E_MINN_train),1,'omitnan')', ...
    mean(abs(E_MINN_test),1,'omitnan')', ...
    mean(abs(SHAP_ANN_train),1,'omitnan')', ...
    mean(abs(SHAP_ANN_test),1,'omitnan')', ...
    mean(abs(SHAP_MINN_train),1,'omitnan')', ...
    mean(abs(SHAP_MINN_test),1,'omitnan')', ...
    'VariableNames', { ...
    'Feature', ...
    'ANN_Train_MeanAbsSensitivity', ...
    'ANN_Test_MeanAbsSensitivity', ...
    'MINN_Train_MeanAbsSensitivity', ...
    'MINN_Test_MeanAbsSensitivity', ...
    'ANN_Train_MeanAbsOutputChangePercent', ...
    'ANN_Test_MeanAbsOutputChangePercent', ...
    'MINN_Train_MeanAbsOutputChangePercent', ...
    'MINN_Test_MeanAbsOutputChangePercent', ...
    'ANN_Train_MeanAbsElasticity', ...
    'ANN_Test_MeanAbsElasticity', ...
    'MINN_Train_MeanAbsElasticity', ...
    'MINN_Test_MeanAbsElasticity', ...
    'ANN_Train_MeanAbsSHAP', ...
    'ANN_Test_MeanAbsSHAP', ...
    'MINN_Train_MeanAbsSHAP', ...
    'MINN_Test_MeanAbsSHAP' ...
    });

%% ============================================================
% EXPORT TO EXCEL
%% ============================================================

if isfile(outExcel)
    delete(outExcel);
end

writetable(All_Sen_Elasticity, outExcel, 'Sheet','All_Sen_Elasticity');
writetable(All_SHAP_Summary,  outExcel, 'Sheet','All_SHAP_Summary');
writetable(All_Summary,       outExcel, 'Sheet','All_Summary');

writetable(Tpred_ANN_train,  outExcel, 'Sheet','ANN_Train');
writetable(Tpred_ANN_test,   outExcel, 'Sheet','ANN_Test');
writetable(Tpred_MINN_train, outExcel, 'Sheet','MINN_Train');
writetable(Tpred_MINN_test,  outExcel, 'Sheet','MINN_Test');

writetable(ANN_SenElas_Train,  outExcel, 'Sheet','ANN_SenElas_Train');
writetable(ANN_SenElas_Test,   outExcel, 'Sheet','ANN_SenElas_Test');
writetable(MINN_SenElas_Train, outExcel, 'Sheet','MINN_SenElas_Train');
writetable(MINN_SenElas_Test,  outExcel, 'Sheet','MINN_SenElas_Test');

writetable(Sensitivity_ANN_Train,  outExcel, 'Sheet','ANN_Sensitivity_Train');
writetable(Sensitivity_ANN_Test,   outExcel, 'Sheet','ANN_Sensitivity_Test');
writetable(Sensitivity_MINN_Train, outExcel, 'Sheet','MINN_Sensitivity_Train');
writetable(Sensitivity_MINN_Test,  outExcel, 'Sheet','MINN_Sensitivity_Test');

writetable(OutputChange_ANN_Train,  outExcel, 'Sheet','ANN_OutputChange_Train');
writetable(OutputChange_ANN_Test,   outExcel, 'Sheet','ANN_OutputChange_Test');
writetable(OutputChange_MINN_Train, outExcel, 'Sheet','MINN_OutputChange_Train');
writetable(OutputChange_MINN_Test,  outExcel, 'Sheet','MINN_OutputChange_Test');

writetable(Elasticity_ANN_Train,  outExcel, 'Sheet','ANN_Elasticity_Train');
writetable(Elasticity_ANN_Test,   outExcel, 'Sheet','ANN_Elasticity_Test');
writetable(Elasticity_MINN_Train, outExcel, 'Sheet','MINN_Elasticity_Train');
writetable(Elasticity_MINN_Test,  outExcel, 'Sheet','MINN_Elasticity_Test');

writetable(SHAP_ANN_Train_Table,  outExcel, 'Sheet','ANN_SHAP_Train');
writetable(SHAP_ANN_Test_Table,   outExcel, 'Sheet','ANN_SHAP_Test');
writetable(SHAP_MINN_Train_Table, outExcel, 'Sheet','MINN_SHAP_Train');
writetable(SHAP_MINN_Test_Table,  outExcel, 'Sheet','MINN_SHAP_Test');

fprintf('\n✅ Exported Excel Successfully: %s\n', outExcel);

%% ============================================================
% FIGURES
%% ============================================================

plotGroupedBar(featShow, ...
    mean(abs(E_ANN_train),1,'omitnan')', ...
    mean(abs(E_MINN_train),1,'omitnan')', ...
    'Mean Absolute Elasticity of Training Data', ...
    'Mean absolute elasticity', ...
    'Train_MeanAbsElasticity_ANN_MINN.png', ...
    fontName, fontSize, outDPI, doExportFigure,outDir);

plotGroupedBar(featShow, ...
    mean(abs(E_ANN_test),1,'omitnan')', ...
    mean(abs(E_MINN_test),1,'omitnan')', ...
    'Mean Absolute Elasticity of Testing Data', ...
    'Mean absolute elasticity', ...
    'Test_MeanAbsElasticity_ANN_MINN.png', ...
    fontName, fontSize, outDPI, doExportFigure,outDir);

plotGroupedBar(featShow, ...
    mean(abs(SHAP_ANN_train),1,'omitnan')', ...
    mean(abs(SHAP_MINN_train),1,'omitnan')', ...
    'SHAP Importance of Training Data', ...
    'Mean absolute SHAP value', ...
    'Train_SHAP_ANN_MINN.png', ...
    fontName, fontSize, outDPI, doExportFigure,outDir);

plotGroupedBar(featShow, ...
    mean(abs(SHAP_ANN_test),1,'omitnan')', ...
    mean(abs(SHAP_MINN_test),1,'omitnan')', ...
    'SHAP Importance of Testing Data', ...
    'Mean absolute SHAP value', ...
    'Test_SHAP_ANN_MINN.png', ...
    fontName, fontSize, outDPI, doExportFigure,outDir);

disp('Done.');

%% ============================================================
% LOCAL FUNCTIONS
%% ============================================================

function [T,selectedSheet] = readDataSheetAuto(excelFile)
if ~isfile(excelFile)
    [selectedFile,selectedFolder]=uigetfile({'*.xlsx;*.xls','Excel files'}, ...
        sprintf('Select %s',excelFile));
    if isequal(selectedFile,0), error('Excel data file was not selected.'); end
    excelFile=fullfile(selectedFolder,selectedFile);
end
try
    sheets=sheetnames(excelFile);
catch
    [~,sheetCell]=xlsfinfo(excelFile); sheets=string(sheetCell);
end
T=table; selectedSheet=''; reports=strings(numel(sheets),1);
for iSheet=1:numel(sheets)
    try
        C=readtable(excelFile,'Sheet',sheets(iSheet),'VariableNamingRule','preserve');
        C.Properties.VariableNames=matlab.lang.makeValidName( ...
            strtrim(C.Properties.VariableNames));
        V=string(C.Properties.VariableNames); reports(iSheet)=strjoin(V,', ');
        hasD=any(strcmpi(V,'D')); hasT=any(strcmpi(V,'t'));
        hasFy=any(strcmpi(V,'fy'));
        hasFc=any(strcmpi(V,'fc')) || any(strcmpi(V,'fcu'));
        hasN=any(strcmpi(V,'N'));
        hasLD=any(V=="L_D") || any(V=="LD") || any(V=="LoverD") || ...
            any(startsWith(V,"L_D"));
        if hasD && hasT && hasFy && hasFc && hasN && hasLD
            T=C; selectedSheet=char(sheets(iSheet)); return;
        end
    catch ME
        reports(iSheet)="Read error: "+string(ME.message);
    end
end
details=strings(numel(sheets),1);
for iSheet=1:numel(sheets)
    details(iSheet)=string(sheets(iSheet))+": "+reports(iSheet);
end
error(['No worksheet contains D, t, fy/Fy, fc/Fc, L/D, and N. ' ...
    'Worksheets inspected:\n%s'],strjoin(details,newline));
end

function rootFolder=locateFiveFoldRoot(folderName,modelPrefix,scriptFolder)
candidates={folderName,fullfile(pwd,folderName),fullfile(scriptFolder,folderName)};
rootFolder='';
for i=1:numel(candidates)
    if hasFiveFoldModels(candidates{i},modelPrefix)
        rootFolder=candidates{i}; return;
    end
end
selectedFolder=uigetdir(scriptFolder,sprintf( ...
    'Select %s containing five %s fold models',folderName,modelPrefix));
if isequal(selectedFolder,0), error('%s model folder was not selected.',modelPrefix); end
if ~hasFiveFoldModels(selectedFolder,modelPrefix)
    error('Selected folder does not contain all five %s models.',modelPrefix);
end
rootFolder=selectedFolder;
end

function tf=hasFiveFoldModels(rootFolder,modelPrefix)
tf=isfolder(rootFolder); if ~tf, return; end
for fold=1:5
    foldFolder=fullfile(rootFolder,sprintf('Fold_%02d',fold));
    pattern=sprintf('%s_Model_Fold_%02d*.mat',modelPrefix,fold);
    if ~isfolder(foldFolder) || isempty(dir(fullfile(foldFolder,pattern)))
        tf=false; return;
    end
end
end

function modelFile=selectFoldModelFile(rootFolder,modelPrefix,fold)
foldFolder=fullfile(rootFolder,sprintf('Fold_%02d',fold));
pattern=sprintf('%s_Model_Fold_%02d*.mat',modelPrefix,fold);
matches=dir(fullfile(foldFolder,pattern));
if isempty(matches), error('%s Fold %d model was not found.',modelPrefix,fold); end
if numel(matches)>1
    [~,idxNewest]=max([matches.datenum]); matches=matches(idxNewest);
    warning('Multiple %s Fold %d files found; using %s.', ...
        modelPrefix,fold,matches.name);
end
modelFile=fullfile(matches.folder,matches.name);
end

function checkModelFields(S, modelName)

    if ~isfield(S,'net')
        error('%s model file has no variable "net".', modelName);
    end

    requiredFields = {'muX','stdX','muy','stdy'};

    for k = 1:numel(requiredFields)
        if ~isfield(S, requiredFields{k})
            error('%s model file missing variable "%s".', ...
                modelName, requiredFields{k});
        end
    end
end

function [tblX, featShow] = buildFeatureTable_auto(T, nFeat)

    V = string(T.Properties.VariableNames);

    if ~any(V=="D")
        error('Missing column D');
    end

    if ~any(V=="t")
        error('Missing column t');
    end

    D = T.D;
    t = T.t;

    if any(V=="Fy")
        Fy = T.Fy;
    elseif any(V=="fy")
        Fy = T.fy;
    else
        error('Missing Fy or fy column');
    end

    if any(V=="Fc")
        Fc = T.Fc;
    elseif any(V=="fc")
        Fc = T.fc;
    elseif any(V=="fcu")
        Fc = T.fcu;
    else
        error('Missing Fc, fc, or fcu column');
    end

    if any(V=="L_D")
        LD = T.L_D;
    elseif any(V=="LD")
        LD = T.LD;
    elseif any(V=="LoverD")
        LD = T.LoverD;
    elseif any(V=="L_over_D")
        LD = T.L_over_D;
    else
        error('Missing L/D column.');
    end

    if nFeat == 5
        tblX = table(D, t, Fy, Fc, LD, ...
            'VariableNames', {'D','t','Fy','Fc','L_D'});

        featShow = {'D','t','f_y','f_c','L/D'};
        return
    end

    if any(V=="e_r0")
        er = T.e_r0;
    elseif any(V=="e_r")
        er = T.e_r;
    elseif any(V=="er")
        er = T.er;
    elseif any(V=="E_R0")
        er = T.E_R0;
    elseif any(V=="E_R")
        er = T.E_R;
    else
        error('nFeat = 6 requires e/r column.');
    end

    tblX = table(D, t, Fy, Fc, LD, er, ...
        'VariableNames', {'D','t','Fy','Fc','L_D','e_r0'});

    featShow = {'D','t','f_y','f_c','L/D','e/r'};
end

function [E, dY, percentChange] = calculateElasticityForward( ...
    X, net, muX, stdX, muy, stdy, relStep_default)

    n = size(X,1);
    nFeat = size(X,2);

    E = nan(n,nFeat);
    dY = nan(n,nFeat);
    percentChange = nan(n,nFeat);

    Y0 = predictModel_vec(X, net, muX, stdX, muy, stdy);

    fprintf('Predicted output range: %.4f to %.4f\n', ...
        min(Y0,[],'omitnan'), max(Y0,[],'omitnan'));

    for i = 1:n

        xi = X(i,:);
        f0 = Y0(i);

        if ~isfinite(f0) || f0 == 0
            continue
        end

        for j = 1:nFeat

            delta_x = xi(j) * relStep_default;

            if delta_x == 0 || ~isfinite(delta_x)
                continue
            end

            x_plus = xi;
            x_plus(j) = xi(j) * (1 + relStep_default);

            if nFeat >= 2
                if x_plus(1) <= 2*x_plus(2)
                    x_plus(1) = 2*x_plus(2) + 1e-6;
                end
            end

            f_plus = predictModel_single( ...
                x_plus, net, muX, stdX, muy, stdy);

            if ~isfinite(f_plus)
                continue
            end

            delta_y = f_plus - f0;

            dY(i,j) = delta_y / delta_x;

            percentChange(i,j) = (delta_y / f0) * 100;

            E(i,j) = percentChange(i,j) / (relStep_default * 100);
        end
    end
end

function Tsummary = createSenElasSummary( ...
    featShow, dY, percentChange, E, modelName, dataName, relStep_default)

    Tsummary = table( ...
        repmat({modelName}, numel(featShow), 1), ...
        repmat({dataName}, numel(featShow), 1), ...
        featShow(:), ...
        repmat(relStep_default*100, numel(featShow), 1), ...
        mean(dY,1,'omitnan')', ...
        median(dY,1,'omitnan')', ...
        std(dY,0,1,'omitnan')', ...
        min(dY,[],1,'omitnan')', ...
        max(dY,[],1,'omitnan')', ...
        mean(percentChange,1,'omitnan')', ...
        median(percentChange,1,'omitnan')', ...
        std(percentChange,0,1,'omitnan')', ...
        min(percentChange,[],1,'omitnan')', ...
        max(percentChange,[],1,'omitnan')', ...
        mean(E,1,'omitnan')', ...
        median(E,1,'omitnan')', ...
        std(E,0,1,'omitnan')', ...
        min(E,[],1,'omitnan')', ...
        max(E,[],1,'omitnan')', ...
        abs(mean(dY,1,'omitnan'))', ...
        abs(mean(E,1,'omitnan'))', ...
        'VariableNames', { ...
        'Model', ...
        'Data', ...
        'Variable', ...
        'Input_Change_%', ...
        'Mean_Sensitivity', ...
        'Median_Sensitivity', ...
        'Std_Sensitivity', ...
        'Min_Sensitivity', ...
        'Max_Sensitivity', ...
        'Mean_Output_Change_%', ...
        'Median_Output_Change_%', ...
        'Std_Output_Change_%', ...
        'Min_Output_Change_%', ...
        'Max_Output_Change_%', ...
        'Mean_Elasticity', ...
        'Median_Elasticity', ...
        'Std_Elasticity', ...
        'Min_Elasticity', ...
        'Max_Elasticity', ...
        'Abs_Mean_Sensitivity', ...
        'Abs_Mean_Elasticity' ...
        });

    Tsummary.Sensitivity_Rank = tiedrank(-Tsummary.Abs_Mean_Sensitivity);
    Tsummary.Elasticity_Rank  = tiedrank(-Tsummary.Abs_Mean_Elasticity);

    Tsummary = sortrows(Tsummary, 'Abs_Mean_Elasticity', 'descend');
end

function Tshap = createSHAPSummary(featShow, SHAP, modelName, dataName)

    Tshap = table( ...
        repmat({modelName}, numel(featShow), 1), ...
        repmat({dataName}, numel(featShow), 1), ...
        featShow(:), ...
        mean(SHAP,1,'omitnan')', ...
        mean(abs(SHAP),1,'omitnan')', ...
        std(SHAP,0,1,'omitnan')', ...
        min(SHAP,[],1,'omitnan')', ...
        max(SHAP,[],1,'omitnan')', ...
        'VariableNames', { ...
        'Model', ...
        'Data', ...
        'Variable', ...
        'Mean_SHAP', ...
        'Mean_Abs_SHAP', ...
        'Std_SHAP', ...
        'Min_SHAP', ...
        'Max_SHAP' ...
        });

    Tshap.SHAP_Rank = tiedrank(-Tshap.Mean_Abs_SHAP);

    Tshap = sortrows(Tshap, 'Mean_Abs_SHAP', 'descend');
end

function SHAP = calculatePermutationSHAP( ...
    Xtarget, Xbackground, net, muX, stdX, muy, stdy, nPerm)

    nSample = size(Xtarget,1);
    nFeat = size(Xtarget,2);

    SHAP = nan(nSample, nFeat);

    basePred = predictModel_vec(Xbackground, net, muX, stdX, muy, stdy);
    baseline = mean(basePred, 'omitnan');

    fprintf('Baseline prediction = %.4f\n', baseline);

    for i = 1:nSample

        x = Xtarget(i,:);
        phi = zeros(1,nFeat);

        for p = 1:nPerm

            order = randperm(nFeat);

            xb = Xbackground(randi(size(Xbackground,1)), :);

            x_prev = xb;
            f_prev = predictModel_single(x_prev, net, muX, stdX, muy, stdy);

            for k = 1:nFeat

                j = order(k);

                x_new = x_prev;
                x_new(j) = x(j);

                f_new = predictModel_single(x_new, net, muX, stdX, muy, stdy);

                phi(j) = phi(j) + (f_new - f_prev);

                x_prev = x_new;
                f_prev = f_new;
            end
        end

        SHAP(i,:) = phi / nPerm;

        if mod(i,20) == 0
            fprintf('SHAP progress: %d/%d\n', i, nSample);
        end
    end
end

function y = predictModel_vec(X, net, muX, stdX, muy, stdy)

    n = size(X,1);
    y = nan(n,1);

    for i = 1:n
        y(i) = predictModel_single(X(i,:), net, muX, stdX, muy, stdy);
    end
end

function y = predictModel_single(x, net, muX, stdX, muy, stdy)

    x = double(x(:))';

    xn = (x - muX.') ./ stdX.';

    try
        if isa(net,'dlnetwork')
            dlX = dlarray(xn','CB');
            dlY = forward(net, dlX);
            yhat_n = extractdata(dlY)';
        else
            yhat_n = predict(net, xn);
        end
    catch
        yhat_n = predict(net, xn);
    end

    y = double(yhat_n) * stdy + muy;
end

function y=predictANNEnsemble_vec(X,A)
    K=numel(A.bestFoldModels); P=nan(size(X,1),K);
    for f=1:K
        P(:,f)=predictModel_vec(X,A.bestFoldModels{f}, ...
            A.bestFoldMuX{f}(:),A.bestFoldStdX{f}(:), ...
            A.bestFoldMuy(f),A.bestFoldStdy(f));
    end
    y=mean(P,2,'omitnan');
end

function y=predictANNEnsemble_single(x,A)
    K=numel(A.bestFoldModels); P=nan(K,1);
    for f=1:K
        P(f)=predictModel_single(x,A.bestFoldModels{f}, ...
            A.bestFoldMuX{f}(:),A.bestFoldStdX{f}(:), ...
            A.bestFoldMuy(f),A.bestFoldStdy(f));
    end
    y=mean(P,'omitnan');
end

function y=predictMINNEnsemble_vec(X,M,N)
    K=numel(M); P=nan(size(X,1),K);
    for f=1:K
        q=N{f}; P(:,f)=predictModel_vec(X,M{f},q.muX(:), ...
            q.stdX(:),q.muy,q.stdy);
    end
    y=mean(P,2,'omitnan');
end

function y=predictMINNEnsemble_single(x,M,N)
    K=numel(M); P=nan(K,1);
    for f=1:K
        q=N{f}; P(f)=predictModel_single(x,M{f},q.muX(:), ...
            q.stdX(:),q.muy,q.stdy);
    end
    y=mean(P,'omitnan');
end

function [E,dY,pct]=calculateElasticityForward_ANNEnsemble(X,A,step)
    fVec=@(Z) predictANNEnsemble_vec(Z,A);
    fOne=@(z) predictANNEnsemble_single(z,A);
    [E,dY,pct]=calculateElasticityWithPredictors(X,fVec,fOne,step,'ANN');
end

function [E,dY,pct]=calculateElasticityForward_MINNEnsemble(X,M,N,step)
    fVec=@(Z) predictMINNEnsemble_vec(Z,M,N);
    fOne=@(z) predictMINNEnsemble_single(z,M,N);
    [E,dY,pct]=calculateElasticityWithPredictors(X,fVec,fOne,step,'MINN');
end

function [E,dY,pct]=calculateElasticityWithPredictors(X,fVec,fOne,step,label)
    n=size(X,1); p=size(X,2); E=nan(n,p); dY=nan(n,p); pct=nan(n,p);
    Y0=fVec(X);
    fprintf('%s ensemble output range: %.4f to %.4f\n',label, ...
        min(Y0,[],'omitnan'),max(Y0,[],'omitnan'));
    for i=1:n
        x=X(i,:); f0=Y0(i); if ~isfinite(f0)||f0==0,continue;end
        for j=1:p
            dx=x(j)*step; if ~isfinite(dx)||dx==0,continue;end
            xp=x; xp(j)=x(j)*(1+step);
            if p>=2 && xp(1)<=2*xp(2),xp(1)=2*xp(2)+1e-6;end
            fp=fOne(xp); if ~isfinite(fp),continue;end
            dy=fp-f0; dY(i,j)=dy/dx; pct(i,j)=100*dy/f0;
            E(i,j)=pct(i,j)/(100*step);
        end
    end
end

function S=calculatePermutationSHAP_ANNEnsemble(X,B,A,nPerm)
    fVec=@(Z) predictANNEnsemble_vec(Z,A);
    fOne=@(z) predictANNEnsemble_single(z,A);
    S=calculatePermutationSHAPWithPredictors(X,B,fVec,fOne,nPerm,'ANN');
end

function S=calculatePermutationSHAP_MINNEnsemble(X,B,M,N,nPerm)
    fVec=@(Z) predictMINNEnsemble_vec(Z,M,N);
    fOne=@(z) predictMINNEnsemble_single(z,M,N);
    S=calculatePermutationSHAPWithPredictors(X,B,fVec,fOne,nPerm,'MINN');
end

function S=calculatePermutationSHAPWithPredictors(X,B,fVec,fOne,nPerm,label)
    n=size(X,1); p=size(X,2); S=nan(n,p);
    fprintf('%s ensemble baseline prediction = %.4f\n',label, ...
        mean(fVec(B),'omitnan'));
    for i=1:n
        phi=zeros(1,p); used=zeros(1,p); x=X(i,:);
        for r=1:nPerm
            order=randperm(p); prev=B(randi(size(B,1)),:); fp=fOne(prev);
            if ~isfinite(fp),continue;end
            for k=1:p
                j=order(k); next=prev; next(j)=x(j); fn=fOne(next);
                if isfinite(fn)&&isfinite(fp)
                    phi(j)=phi(j)+fn-fp; used(j)=used(j)+1;
                end
                prev=next; fp=fn;
            end
        end
        S(i,:)=phi./max(used,1);
        if mod(i,20)==0,fprintf('%s ensemble SHAP: %d/%d\n',label,i,n);end
    end
end

function plotGroupedBar(featShow, y1, y2, figTitle, ...
    yLabelText, outName, ...
    fontName, fontSize, outDPI, doExportFigure, outDir)

    figure('Color','w');
    set(gcf,'Units','centimeters','Position',[5 5 18 10]);

    Xbar = [y1, y2];

    bar(Xbar,'grouped');

    set(gca,'FontName',fontName,'FontSize',fontSize);
    xticks(1:numel(featShow));
    xticklabels(featShow);

    ylabel(yLabelText, ...
        'FontName',fontName,'FontSize',fontSize);

    xlabel('Input variables', ...
        'FontName',fontName,'FontSize',fontSize);

    title(figTitle, ...
        'FontName',fontName,'FontSize',fontSize);

    legend({'ANN','MINN'}, ...
        'Location','northwest', ...
        'FontName',fontName, ...
        'FontSize',fontSize);

    grid on;
    box off;
    set(gca,'TickDir','out');

    if doExportFigure
        exportgraphics(gcf, fullfile(outDir,outName), ...
    'Resolution', outDPI);
    end
end

function checkANNEnsembleFields(S)
    req = {'bestFoldModels','bestFoldMuX','bestFoldStdX', ...
        'bestFoldMuy','bestFoldStdy'};
    for k=1:numel(req)
        if ~isfield(S,req{k}), error('ANN ensemble missing "%s".',req{k}); end
    end
    K=numel(S.bestFoldModels);
    if ~iscell(S.bestFoldModels) || K==0 || ~iscell(S.bestFoldMuX) || ...
            ~iscell(S.bestFoldStdX) || numel(S.bestFoldMuX)~=K || ...
            numel(S.bestFoldStdX)~=K || numel(S.bestFoldMuy)~=K || ...
            numel(S.bestFoldStdy)~=K
        error('ANN models and normalization arrays have inconsistent sizes.');
    end
end

function checkMINNEnsembleFields(S)
    if ~isfield(S,'bestModels') || ~isfield(S,'bestNorm')
        error('MINN result file must contain bestModels and bestNorm.');
    end
    if ~iscell(S.bestModels) || ~iscell(S.bestNorm) || ...
            isempty(S.bestModels) || numel(S.bestModels)~=numel(S.bestNorm)
        error('MINN bestModels and bestNorm have inconsistent sizes.');
    end
end
