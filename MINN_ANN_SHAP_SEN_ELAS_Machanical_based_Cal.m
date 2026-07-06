%% ============================================================
% Yu vs Liang vs ANN vs MINN
% Sensitivity / Elasticity / SHAP comparison
% TESTING DATA ONLY
%
% IMPORTANT:
%   1) Read testing data only.
%   2) First filter specimens satisfying BOTH Yu and Liang limitations.
%   3) Use the same filtered testing specimens for Yu, Liang, ANN, and MINN.
%   4) No grouped bar charts.
%   5) Plot Input value vs Sensitivity / Elasticity / SHAP using 2x2 tiles.
%
% Required files in the same MATLAB folder:
%   - cfst_model_N_S_ANN.mat
%   - cfst_model_N_S_MINN_test.mat
%   - N_S_Testing_DATASET_25.xlsx
%
% Output:
%   - Yu_Liang_ANN_MINN_SHAP_SEN_ELAS_TestOnly_Filtered.xlsx
%   - Figures in folder: Figures_Yu_Liang_ANN_MINN_TestOnly_Filtered
%% ============================================================

clear; clc; close all;

%% ---------------- USER SETTINGS ----------------
annModelFile  = 'cfst_model_N_S_ANN.mat';
minnModelFile = 'cfst_model_N_S_MINN_ver2.mat';

testFile  = 'N_S_Testing_DATASET_25.xlsx';

outExcel = 'SHAP_SEN_ELAS_Machanical_based_compard.xlsx';
outDir   = 'SHAP_SEN_ELAS_Machanical_based_compard';

fontName = 'Times New Roman';
fontSize = 12;
outDPI = 1200;
doExportFigure = true;

% Forward perturbation: +1%
relStep_default = 0.01;

% Permutation SHAP
% ถ้ารันช้า ลดเป็น 30 หรือ 50 ได้
nPermSHAP = 100;

rng(500);

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% ---------------- LOAD TESTING DATA ONLY ----------------
Tte = readtable(testFile,'VariableNamingRule','preserve');
Tte.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Tte.Properties.VariableNames));

%% ============================================================
% STEP 1: FILTER TESTING DATA BY YU + LIANG LIMITATIONS FIRST
%% ============================================================
[validYu, validLiang, validCompare, FilterTable] = getYuLiangValidMask(Tte);

nTotal = height(Tte);
nYu = sum(validYu);
nLiang = sum(validLiang);
nCommon = sum(validCompare);

fprintf('\n====================================================\n');
fprintf('TESTING DATA FILTER SUMMARY\n');
fprintf('Original testing data      = %d\n', nTotal);
fprintf('Valid Yu data              = %d\n', nYu);
fprintf('Valid Liang data           = %d\n', nLiang);
fprintf('Valid both Yu and Liang    = %d\n', nCommon);
fprintf('Removed data               = %d\n', nTotal - nCommon);
fprintf('====================================================\n');

if nCommon == 0
    error('No testing specimens satisfy both Yu and Liang limitations.');
end

% Keep only data satisfying both Yu and Liang limitations
Tte_all = Tte;
Tte = Tte(validCompare,:);
FilterTable.Filter_Status = repmat("Removed", nTotal, 1);
FilterTable.Filter_Status(validCompare) = "Used";

%% ---------------- LOAD MODELS ----------------
ANN  = load(annModelFile);
MINN = load(minnModelFile);

checkModelFields(ANN,  'ANN');
checkModelFields(MINN, 'MINN');

nFeat_ANN  = numel(ANN.muX);
nFeat_MINN = numel(MINN.muX);

if nFeat_ANN ~= nFeat_MINN
    error('ANN and MINN have different numbers of input features.');
end

nFeat = nFeat_ANN;

if ~(nFeat == 5 || nFeat == 6)
    error('This script supports only 5-input or 6-input models.');
end

fprintf('Number of input features = %d\n', nFeat);

%% ---------------- BUILD FILTERED TEST INPUT TABLE ----------------
[tblX_test, featShow] = buildFeatureTable_auto(Tte, nFeat);

% Remove missing rows after filtering, if any
idxTestOK = all(~ismissing(tblX_test),2);

if any(~idxTestOK)
    fprintf('Rows removed due to missing input values after Yu/Liang filter = %d\n', sum(~idxTestOK));
end

Tte = Tte(idxTestOK,:);
tblX_test = tblX_test(idxTestOK,:);
Xtest = table2array(tblX_test);

fprintf('Final filtered testing rows used = %d\n', size(Xtest,1));

disp('Feature order used:');
disp(tblX_test.Properties.VariableNames);

%% ============================================================
% STEP 2: PREDICTION ON FILTERED TESTING DATA ONLY
%% ============================================================
fprintf('\nPredicting outputs from Yu, Liang, ANN, and MINN...\n');

Y_Yu_test    = predictFormula_vec(Xtest, 'Yu');
Y_Liang_test = predictFormula_vec(Xtest, 'Liang');

Y_ANN_test = predictModel_vec(Xtest, ANN.net, ANN.muX(:), ANN.stdX(:), ANN.muy, ANN.stdy);
Y_MINN_test = predictModel_vec(Xtest, MINN.net, MINN.muX(:), MINN.stdX(:), MINN.muy, MINN.stdy);

%% ============================================================
% STEP 3: SENSITIVITY + ELASTICITY USING FORWARD +1%
%% ============================================================
fprintf('\nCalculating Yu sensitivity and elasticity...\n');
[E_Yu_test, dY_Yu_test, pct_Yu_test] = calculateElasticityForward_formula(Xtest, 'Yu', relStep_default);

fprintf('\nCalculating Liang sensitivity and elasticity...\n');
[E_Liang_test, dY_Liang_test, pct_Liang_test] = calculateElasticityForward_formula(Xtest, 'Liang', relStep_default);

fprintf('\nCalculating ANN sensitivity and elasticity...\n');
[E_ANN_test, dY_ANN_test, pct_ANN_test] = calculateElasticityForward_model(Xtest, ANN.net, ANN.muX(:), ANN.stdX(:), ANN.muy, ANN.stdy, relStep_default);

fprintf('\nCalculating MINN sensitivity and elasticity...\n');
[E_MINN_test, dY_MINN_test, pct_MINN_test] = calculateElasticityForward_model(Xtest, MINN.net, MINN.muX(:), MINN.stdX(:), MINN.muy, MINN.stdy, relStep_default);

%% ============================================================
% STEP 4: PERMUTATION SHAP ON FILTERED TESTING DATA ONLY
%% ============================================================
% Background = filtered testing data itself
Xbackground = Xtest;

fprintf('\nCalculating Yu SHAP...\n');
SHAP_Yu_test = calculatePermutationSHAP_formula(Xtest, Xbackground, 'Yu', nPermSHAP);

fprintf('\nCalculating Liang SHAP...\n');
SHAP_Liang_test = calculatePermutationSHAP_formula(Xtest, Xbackground, 'Liang', nPermSHAP);

fprintf('\nCalculating ANN SHAP...\n');
SHAP_ANN_test = calculatePermutationSHAP_model(Xtest, Xbackground, ANN.net, ANN.muX(:), ANN.stdX(:), ANN.muy, ANN.stdy, nPermSHAP);

fprintf('\nCalculating MINN SHAP...\n');
SHAP_MINN_test = calculatePermutationSHAP_model(Xtest, Xbackground, MINN.net, MINN.muX(:), MINN.stdX(:), MINN.muy, MINN.stdy, nPermSHAP);

%% ============================================================
% STEP 5: TABLES
%% ============================================================

Pred_Test = tblX_test;
Pred_Test.N_Yu    = Y_Yu_test;
Pred_Test.N_Liang = Y_Liang_test;
Pred_Test.N_ANN   = Y_ANN_test;
Pred_Test.N_MINN  = Y_MINN_test;

if any(strcmp(Tte.Properties.VariableNames,'N'))
    Pred_Test.N_exp = Tte.N;
elseif any(strcmp(Tte.Properties.VariableNames,'Nu'))
    Pred_Test.N_exp = Tte.Nu;
end

Valid_Data_Summary = table( ...
    nTotal, nYu, nLiang, nCommon, nTotal-nCommon, size(Xtest,1), ...
    'VariableNames', {'Original_Testing_Data','Valid_Yu','Valid_Liang','Valid_Both_Yu_Liang','Removed_Data','Final_Used_Data'});

All_Sen_Elasticity = [
    createSenElasSummary(featShow, dY_Yu_test,    pct_Yu_test,    E_Yu_test,    'Yu',    'Test', relStep_default);
    createSenElasSummary(featShow, dY_Liang_test, pct_Liang_test, E_Liang_test, 'Liang', 'Test', relStep_default);
    createSenElasSummary(featShow, dY_ANN_test,   pct_ANN_test,   E_ANN_test,   'ANN',   'Test', relStep_default);
    createSenElasSummary(featShow, dY_MINN_test,  pct_MINN_test,  E_MINN_test,  'MINN',  'Test', relStep_default)
];

All_SHAP_Summary = [
    createSHAPSummary(featShow, SHAP_Yu_test,    'Yu',    'Test');
    createSHAPSummary(featShow, SHAP_Liang_test, 'Liang', 'Test');
    createSHAPSummary(featShow, SHAP_ANN_test,   'ANN',   'Test');
    createSHAPSummary(featShow, SHAP_MINN_test,  'MINN',  'Test')
];

All_Summary = createAllSummary_TestOnly(featShow, ...
    dY_Yu_test, dY_Liang_test, dY_ANN_test, dY_MINN_test, ...
    E_Yu_test,  E_Liang_test,  E_ANN_test,  E_MINN_test, ...
    SHAP_Yu_test, SHAP_Liang_test, SHAP_ANN_test, SHAP_MINN_test);

RankCorrelation_SHAP = createRankCorrelationTable(All_SHAP_Summary, 'SHAP_Rank');
RankCorrelation_Elasticity = createRankCorrelationTable(All_Sen_Elasticity, 'Elasticity_Rank');

%% RAW TABLES
Sensitivity_Yu_Test    = array2table(dY_Yu_test,    'VariableNames', matlab.lang.makeValidName(featShow));
Sensitivity_Liang_Test = array2table(dY_Liang_test, 'VariableNames', matlab.lang.makeValidName(featShow));
Sensitivity_ANN_Test   = array2table(dY_ANN_test,   'VariableNames', matlab.lang.makeValidName(featShow));
Sensitivity_MINN_Test  = array2table(dY_MINN_test,  'VariableNames', matlab.lang.makeValidName(featShow));

Elasticity_Yu_Test    = array2table(E_Yu_test,    'VariableNames', matlab.lang.makeValidName(featShow));
Elasticity_Liang_Test = array2table(E_Liang_test, 'VariableNames', matlab.lang.makeValidName(featShow));
Elasticity_ANN_Test   = array2table(E_ANN_test,   'VariableNames', matlab.lang.makeValidName(featShow));
Elasticity_MINN_Test  = array2table(E_MINN_test,  'VariableNames', matlab.lang.makeValidName(featShow));

SHAP_Yu_Test_Table    = array2table(SHAP_Yu_test,    'VariableNames', matlab.lang.makeValidName(featShow));
SHAP_Liang_Test_Table = array2table(SHAP_Liang_test, 'VariableNames', matlab.lang.makeValidName(featShow));
SHAP_ANN_Test_Table   = array2table(SHAP_ANN_test,   'VariableNames', matlab.lang.makeValidName(featShow));
SHAP_MINN_Test_Table  = array2table(SHAP_MINN_test,  'VariableNames', matlab.lang.makeValidName(featShow));

%% ============================================================
% STEP 6: EXPORT TO EXCEL
%% ============================================================
if isfile(outExcel)
    delete(outExcel);
end

writetable(Valid_Data_Summary, outExcel, 'Sheet','Valid_Data_Summary');
writetable(FilterTable,        outExcel, 'Sheet','Filter_Status_All_Test');
writetable(Pred_Test,          outExcel, 'Sheet','Prediction_Test_Filtered');
writetable(All_Sen_Elasticity, outExcel, 'Sheet','All_Sen_Elasticity');
writetable(All_SHAP_Summary,   outExcel, 'Sheet','All_SHAP_Summary');
writetable(All_Summary,        outExcel, 'Sheet','All_Summary');
writetable(RankCorrelation_SHAP, outExcel, 'Sheet','RankCorr_SHAP');
writetable(RankCorrelation_Elasticity, outExcel, 'Sheet','RankCorr_Elasticity');

writetable(Sensitivity_Yu_Test,    outExcel, 'Sheet','Yu_Sensitivity_Test');
writetable(Sensitivity_Liang_Test, outExcel, 'Sheet','Liang_Sensitivity_Test');
writetable(Sensitivity_ANN_Test,   outExcel, 'Sheet','ANN_Sensitivity_Test');
writetable(Sensitivity_MINN_Test,  outExcel, 'Sheet','MINN_Sensitivity_Test');

writetable(Elasticity_Yu_Test,    outExcel, 'Sheet','Yu_Elasticity_Test');
writetable(Elasticity_Liang_Test, outExcel, 'Sheet','Liang_Elasticity_Test');
writetable(Elasticity_ANN_Test,   outExcel, 'Sheet','ANN_Elasticity_Test');
writetable(Elasticity_MINN_Test,  outExcel, 'Sheet','MINN_Elasticity_Test');

writetable(SHAP_Yu_Test_Table,    outExcel, 'Sheet','Yu_SHAP_Test');
writetable(SHAP_Liang_Test_Table, outExcel, 'Sheet','Liang_SHAP_Test');
writetable(SHAP_ANN_Test_Table,   outExcel, 'Sheet','ANN_SHAP_Test');
writetable(SHAP_MINN_Test_Table,  outExcel, 'Sheet','MINN_SHAP_Test');

fprintf('\nExported Excel Successfully: %s\n', outExcel);

%% ============================================================
% STEP 7: PLOTS - INPUT VALUE VS METRIC, 2x2 MODEL COMPARISON
%% ============================================================
colors.Yu    = [0.10 0.55 0.20];
colors.Liang = [0.85 0.45 0.10];
colors.ANN   = [0.20 0.40 0.90];
colors.MINN  = [0.85 0.20 0.20];

markers.Yu    = 'o';
markers.Liang = '^';
markers.ANN   = 's';
markers.MINN  = 'd';

% Elasticity y-axis setting
% Use VARIABLE-WISE common y-limit:
%   - Each input variable has its own suitable y-scale.
%   - Within the same input variable, Yu, Liang, ANN, and MINN use the same y-limit.
% This makes each elasticity figure readable while keeping model comparison fair.
elasYLims = [];

plotInputVsMetricCompare4( ...
    Xtest, ...
    dY_Yu_test, dY_Liang_test, dY_ANN_test, dY_MINN_test, ...
    featShow, 'Sensitivity', 'Sensitivity', ...
    fullfile(outDir,'Sensitivity'), colors, markers, [], ...
    fontName, outDPI, doExportFigure);

plotInputVsMetricCompare4( ...
    Xtest, ...
    E_Yu_test, E_Liang_test, E_ANN_test, E_MINN_test, ...
    featShow, 'Elasticity (%)', 'Elasticity', ...
    fullfile(outDir,'Elasticity'), colors, markers, elasYLims, ...
    fontName, outDPI, doExportFigure);

% Normalized SHAP contribution (%) for displaying in each SHAP subplot
% Each row = model, each column = input variable
SHAP_Contribution_Percent = [ ...
    calcNormalizedSHAPPercent(SHAP_Yu_test); ...
    calcNormalizedSHAPPercent(SHAP_Liang_test); ...
    calcNormalizedSHAPPercent(SHAP_ANN_test); ...
    calcNormalizedSHAPPercent(SHAP_MINN_test)];

plotInputVsMetricCompare4( ...
    Xtest, ...
    SHAP_Yu_test, SHAP_Liang_test, SHAP_ANN_test, SHAP_MINN_test, ...
    featShow, 'SHAP value', 'SHAP Value', ...
    fullfile(outDir,'SHAP'), colors, markers, [], ...
    fontName, outDPI, doExportFigure, SHAP_Contribution_Percent);

plotRankCorrelationHeatmap(RankCorrelation_SHAP, 'SHAP Rank Correlation', ...
    fullfile(outDir,'RankCorr_SHAP_Yu_Liang_ANN_MINN.png'), fontName, fontSize, outDPI, doExportFigure);

fprintf('\nDone.\n');

%% ============================================================
% LOCAL FUNCTIONS
%% ============================================================

function checkModelFields(S, modelName)
    if ~isfield(S,'net')
        error('%s model file has no variable "net".', modelName);
    end
    requiredFields = {'muX','stdX','muy','stdy'};
    for k = 1:numel(requiredFields)
        if ~isfield(S, requiredFields{k})
            error('%s model file missing variable "%s".', modelName, requiredFields{k});
        end
    end
end

function [validYu, validLiang, validCompare, FilterTable] = getYuLiangValidMask(T)
    V = string(T.Properties.VariableNames);

    if ~any(V=="D"), error('Missing column D'); end
    if ~any(V=="t"), error('Missing column t'); end

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

    Di = D - 2.*t;
    As = pi/4 .* (D.^2 - Di.^2);
    Ac = pi/4 .* Di.^2;

    Dt = D ./ t;
    ratio_fc_fy = Fc ./ Fy;

    validYu = ...
        isfinite(D) & isfinite(t) & isfinite(Fy) & isfinite(Fc) & ...
        Fc > 0  & ...
        Fy > 0 &   ...
        Ac > 0 & As > 0;

    validLiang = ...
        isfinite(D) & isfinite(t) & isfinite(Fy) & isfinite(Fc) & ...
        Dt > 0 & Dt <= 150 & ...
        Fc > 0 & Fy > 0 & t > 0 & Di > 0 & ...
        ratio_fc_fy >= 0.04 & ratio_fc_fy <= 0.20;

    validCompare = validYu & validLiang;

    FilterTable = table(D, t, Fy, Fc, Di, Dt, ratio_fc_fy, validYu, validLiang, validCompare, ...
        'VariableNames', {'D','t','Fy','Fc','Di','D_over_t','Fc_over_Fy','Valid_Yu','Valid_Liang','Valid_Both'});
end

function [tblX, featShow] = buildFeatureTable_auto(T, nFeat)
    V = string(T.Properties.VariableNames);

    if ~any(V=="D"), error('Missing column D'); end
    if ~any(V=="t"), error('Missing column t'); end

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
    elseif any(V=="L_D_")
        LD = T.L_D_;
    elseif any(V=="L_D__")
        LD = T.L_D__;
    else
        error('Missing L/D column.');
    end

    if nFeat == 5
        tblX = table(D, t, Fy, Fc, LD, 'VariableNames', {'D','t','Fy','Fc','L_D'});
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

    tblX = table(D, t, Fy, Fc, LD, er, 'VariableNames', {'D','t','Fy','Fc','L_D','e_r0'});
    featShow = {'D','t','f_y','f_c','L/D','e/r'};
end

%% ---------------- Prediction wrappers ----------------

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

function y = predictFormula_vec(X, formulaName)
    n = size(X,1);
    y = nan(n,1);
    for i = 1:n
        y(i) = predictFormula_single(X(i,:), formulaName);
    end
end

function y = predictFormula_single(x, formulaName)
    D  = x(1);
    t  = x(2);
    Fy = x(3);
    Fc = x(4);

    if any(~isfinite([D t Fy Fc])) || D <= 0 || t <= 0 || Fy <= 0 || Fc <= 0
        y = nan;
        return
    end

    switch lower(string(formulaName))
        case "yu"
            y = calcNu_Yu(D,t,Fy,Fc);
        case "liang"
            y = calcNu_Liang(D,t,Fy,Fc);
        otherwise
            error('Unknown formulaName: %s', formulaName);
    end
end

%% ---------------- Yu and Liang formulas ----------------

function N_Yu = calcNu_Yu(D,t,Fy,Fc)
    Di = max(D - 2*t, 0);
    As = pi/4 * (D^2 - Di^2);
    Ac = pi/4 * Di^2;

    validYu = Fc >= 20 && Fc <= 100 && Fy >= 200 && Fy <= 500 && Ac > 0 && As > 0;

    if ~validYu
        N_Yu = nan;
        return
    end

    fck = 0.8 * Fc;
    Asc = As + Ac;
    beta = As / Asc;
    Omega = 1.0;
    xi_sc = (As * Fy) / (Ac * fck);

    etaYu = (Omega * xi_sc) / ...
        ((2.0*Omega + 0.05*xi_sc + (0.2*fck/Fy - 0.05)*xi_sc*Omega) * (Omega + xi_sc));

    fsc = (1 + etaYu) * ((1 - beta)*fck + beta*Fy);
    N_Yu = fsc * Asc / 1000; % kN
end

function N_Liang = calcNu_Liang(D,t,Fy,Fc)
    Di = max(D - 2*t, 0);
    As = pi/4 * (D^2 - Di^2);
    Ac = pi/4 * Di^2;

    Dt = D / t;
    ratio_fc_fy = Fc / Fy;

    validLiang = Dt > 0 && Dt <= 150 && Fc > 0 && Fy > 0 && t > 0 && ...
                 Di > 0 && ratio_fc_fy >= 0.04 && ratio_fc_fy <= 0.20;

    if ~validLiang
        N_Liang = nan;
        return
    end

    Dc = Di;

    gamma_c = 1.85 * Dc^(-0.135);
    gamma_c = min(max(gamma_c,0.85),1.0);

    gamma_s = 1.458 * Dt^(-0.1);
    gamma_s = min(max(gamma_s,0.9),1.1);

    frp = 0;

    if Dt <= 47
        ve0 = 0.881e-6 * Dt^3 - 2.58e-4 * Dt^2 + 1.953e-2 * Dt + 0.4011;

        ve = 0.2312 + 0.3582 * ve0 - 0.1524 * ratio_fc_fy + ...
             4.843 * ve0 * ratio_fc_fy - 9.169 * ratio_fc_fy^2;

        vs = 0.5;
        frp = 0.7 * (ve - vs) * (2*t / (D - 2*t)) * Fy;
        frp = max(frp,0);

    elseif Dt > 47 && Dt <= 150
        frp = (0.006241 - 0.0000357 * Dt) * Fy;
        frp = max(frp,0);
    end

    N_Liang = ((gamma_c * Fc + 4.1 * frp) * Ac + gamma_s * Fy * As) / 1000; % kN
end

%% ---------------- Sensitivity and elasticity ----------------

function [E, dY, percentChange] = calculateElasticityForward_model(X, net, muX, stdX, muy, stdy, relStep_default)
    n = size(X,1);
    nFeat = size(X,2);

    E = nan(n,nFeat);
    dY = nan(n,nFeat);
    percentChange = nan(n,nFeat);

    Y0 = predictModel_vec(X, net, muX, stdX, muy, stdy);

    fprintf('Predicted output range: %.4f to %.4f\n', min(Y0,[],'omitnan'), max(Y0,[],'omitnan'));

    for i = 1:n
        xi = X(i,:);
        f0 = Y0(i);
        if ~isfinite(f0) || f0 == 0
            continue
        end

        for j = 1:nFeat
            [x_plus, delta_x] = perturbPlusOnePercent(xi, j, relStep_default);
            if ~isfinite(delta_x) || delta_x == 0
                continue
            end

            f_plus = predictModel_single(x_plus, net, muX, stdX, muy, stdy);
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

function [E, dY, percentChange] = calculateElasticityForward_formula(X, formulaName, relStep_default)
    n = size(X,1);
    nFeat = size(X,2);

    E = nan(n,nFeat);
    dY = nan(n,nFeat);
    percentChange = nan(n,nFeat);

    Y0 = predictFormula_vec(X, formulaName);

    fprintf('%s output range: %.4f to %.4f\n', formulaName, min(Y0,[],'omitnan'), max(Y0,[],'omitnan'));

    for i = 1:n
        xi = X(i,:);
        f0 = Y0(i);
        if ~isfinite(f0) || f0 == 0
            continue
        end

        for j = 1:nFeat
            [x_plus, delta_x] = perturbPlusOnePercent(xi, j, relStep_default);
            if ~isfinite(delta_x) || delta_x == 0
                continue
            end

            f_plus = predictFormula_single(x_plus, formulaName);
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

function [x_plus, delta_x] = perturbPlusOnePercent(xi, j, relStep)
    x_plus = xi;
    delta_x = xi(j) * relStep;
    x_plus(j) = xi(j) * (1 + relStep);

    % Basic geometry check: D must be greater than 2t
    if numel(x_plus) >= 2
        if x_plus(1) <= 2*x_plus(2)
            x_plus(1) = 2*x_plus(2) + 1e-6;
        end
    end
end

%% ---------------- Permutation SHAP ----------------

function SHAP = calculatePermutationSHAP_model(Xtarget, Xbackground, net, muX, stdX, muy, stdy, nPerm)
    nSample = size(Xtarget,1);
    nFeat = size(Xtarget,2);
    SHAP = nan(nSample, nFeat);

    for i = 1:nSample
        x = Xtarget(i,:);
        phi = zeros(1,nFeat);
        nUsed = zeros(1,nFeat);

        for p = 1:nPerm
            order = randperm(nFeat);
            xb = Xbackground(randi(size(Xbackground,1)), :);

            x_prev = xb;
            f_prev = predictModel_single(x_prev, net, muX, stdX, muy, stdy);
            if ~isfinite(f_prev)
                continue
            end

            for k = 1:nFeat
                j = order(k);
                x_new = x_prev;
                x_new(j) = x(j);

                f_new = predictModel_single(x_new, net, muX, stdX, muy, stdy);
                if isfinite(f_new) && isfinite(f_prev)
                    phi(j) = phi(j) + (f_new - f_prev);
                    nUsed(j) = nUsed(j) + 1;
                end

                x_prev = x_new;
                f_prev = f_new;
            end
        end

        SHAP(i,:) = phi ./ max(nUsed,1);

        if mod(i,20) == 0
            fprintf('SHAP progress: %d/%d\n', i, nSample);
        end
    end
end

function SHAP = calculatePermutationSHAP_formula(Xtarget, Xbackground, formulaName, nPerm)
    nSample = size(Xtarget,1);
    nFeat = size(Xtarget,2);
    SHAP = nan(nSample, nFeat);

    for i = 1:nSample
        x = Xtarget(i,:);
        phi = zeros(1,nFeat);
        nUsed = zeros(1,nFeat);

        for p = 1:nPerm
            order = randperm(nFeat);
            xb = Xbackground(randi(size(Xbackground,1)), :);

            x_prev = xb;
            f_prev = predictFormula_single(x_prev, formulaName);
            if ~isfinite(f_prev)
                continue
            end

            for k = 1:nFeat
                j = order(k);
                x_new = x_prev;
                x_new(j) = x(j);

                f_new = predictFormula_single(x_new, formulaName);
                if isfinite(f_new) && isfinite(f_prev)
                    phi(j) = phi(j) + (f_new - f_prev);
                    nUsed(j) = nUsed(j) + 1;
                end

                x_prev = x_new;
                f_prev = f_new;
            end
        end

        SHAP(i,:) = phi ./ max(nUsed,1);

        if mod(i,20) == 0
            fprintf('%s SHAP progress: %d/%d\n', formulaName, i, nSample);
        end
    end
end

%% ---------------- Summary functions ----------------

function Tsummary = createSenElasSummary(featShow, dY, percentChange, E, modelName, dataName, relStep_default)
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
        mean(abs(dY),1,'omitnan')', ...
        mean(abs(E),1,'omitnan')', ...
        'VariableNames', { ...
        'Model','Data','Variable','Input_Change_percent', ...
        'Mean_Sensitivity','Median_Sensitivity','Std_Sensitivity','Min_Sensitivity','Max_Sensitivity', ...
        'Mean_Output_Change_percent','Median_Output_Change_percent','Std_Output_Change_percent','Min_Output_Change_percent','Max_Output_Change_percent', ...
        'Mean_Elasticity','Median_Elasticity','Std_Elasticity','Min_Elasticity','Max_Elasticity', ...
        'Mean_Abs_Sensitivity','Mean_Abs_Elasticity'} );

    Tsummary.Sensitivity_Rank = tiedrank(-Tsummary.Mean_Abs_Sensitivity);
    Tsummary.Elasticity_Rank  = tiedrank(-Tsummary.Mean_Abs_Elasticity);
    Tsummary = sortrows(Tsummary, {'Model','Data','Elasticity_Rank'});
end

function Tshap = createSHAPSummary(featShow, SHAP, modelName, dataName)
    meanAbsSHAP = mean(abs(SHAP),1,'omitnan')';
    totalAbs = sum(meanAbsSHAP,'omitnan');
    if totalAbs > 0
        normalizedSHAP = meanAbsSHAP ./ totalAbs .* 100;
    else
        normalizedSHAP = nan(size(meanAbsSHAP));
    end

    Tshap = table( ...
        repmat({modelName}, numel(featShow), 1), ...
        repmat({dataName}, numel(featShow), 1), ...
        featShow(:), ...
        mean(SHAP,1,'omitnan')', ...
        meanAbsSHAP, ...
        normalizedSHAP, ...
        std(SHAP,0,1,'omitnan')', ...
        min(SHAP,[],1,'omitnan')', ...
        max(SHAP,[],1,'omitnan')', ...
        'VariableNames', { ...
        'Model','Data','Variable','Mean_SHAP','Mean_Abs_SHAP','Normalized_SHAP_percent','Std_SHAP','Min_SHAP','Max_SHAP'} );

    Tshap.SHAP_Rank = tiedrank(-Tshap.Mean_Abs_SHAP);
    Tshap = sortrows(Tshap, {'Model','Data','SHAP_Rank'});
end


function normalizedSHAP = calcNormalizedSHAPPercent(SHAP)
    % Calculate normalized SHAP contribution (%) from mean absolute SHAP
    % normalized_i = mean(abs(SHAP_i)) / sum(mean(abs(SHAP_all))) * 100
    meanAbsSHAP = mean(abs(SHAP), 1, 'omitnan');
    totalAbs = sum(meanAbsSHAP, 'omitnan');

    if totalAbs > 0
        normalizedSHAP = meanAbsSHAP ./ totalAbs .* 100;
    else
        normalizedSHAP = nan(size(meanAbsSHAP));
    end
end

function All_Summary = createAllSummary_TestOnly(featShow, dY_Yu, dY_Liang, dY_ANN, dY_MINN, E_Yu, E_Liang, E_ANN, E_MINN, SHAP_Yu, SHAP_Liang, SHAP_ANN, SHAP_MINN)
    modelNames = {'Yu','Liang','ANN','MINN'};
    dY_list = {dY_Yu, dY_Liang, dY_ANN, dY_MINN};
    E_list = {E_Yu, E_Liang, E_ANN, E_MINN};
    SHAP_list = {SHAP_Yu, SHAP_Liang, SHAP_ANN, SHAP_MINN};

    rows = cell(numel(modelNames),1);

    for b = 1:numel(modelNames)
        dY = dY_list{b};
        E = E_list{b};
        SHAP = SHAP_list{b};

        rows{b} = table( ...
            repmat(modelNames(b), numel(featShow), 1), ...
            repmat({'Test'}, numel(featShow), 1), ...
            featShow(:), ...
            mean(abs(dY),1,'omitnan')', ...
            mean(abs(E),1,'omitnan')', ...
            mean(abs(SHAP),1,'omitnan')', ...
            'VariableNames', {'Model','Data','Variable','Mean_Abs_Sensitivity','Mean_Abs_Elasticity','Mean_Abs_SHAP'} );
    end

    All_Summary = vertcat(rows{:});
    All_Summary.Elasticity_Rank = nan(height(All_Summary),1);
    All_Summary.SHAP_Rank = nan(height(All_Summary),1);

    for i = 1:numel(modelNames)
        idx = strcmp(All_Summary.Model,modelNames{i});
        All_Summary.Elasticity_Rank(idx) = tiedrank(-All_Summary.Mean_Abs_Elasticity(idx));
        All_Summary.SHAP_Rank(idx) = tiedrank(-All_Summary.Mean_Abs_SHAP(idx));
    end
end

function Tcorr = createRankCorrelationTable(Tsummary, rankVarName)
    models = {'Yu','Liang','ANN','MINN'};

    outModel1 = {};
    outModel2 = {};
    outSpearman = [];

    for i = 1:numel(models)
        for j = i+1:numel(models)
            idx1 = strcmp(Tsummary.Model,models{i});
            idx2 = strcmp(Tsummary.Model,models{j});

            T1 = Tsummary(idx1, {'Variable', rankVarName});
            T2 = Tsummary(idx2, {'Variable', rankVarName});

            [~, ia, ib] = intersect(T1.Variable, T2.Variable, 'stable');
            r1 = T1.(rankVarName)(ia);
            r2 = T2.(rankVarName)(ib);

            ok = isfinite(r1) & isfinite(r2);
            if sum(ok) >= 2
                rho = corr(r1(ok), r2(ok), 'Type','Spearman', 'Rows','complete');
            else
                rho = nan;
            end

            outModel1{end+1,1} = models{i}; %#ok<AGROW>
            outModel2{end+1,1} = models{j}; %#ok<AGROW>
            outSpearman(end+1,1) = rho; %#ok<AGROW>
        end
    end

    Tcorr = table(outModel1, outModel2, outSpearman, ...
        'VariableNames', {'Model_1','Model_2','Spearman_Rank_Correlation'});
end

%% ---------------- Plotting ----------------

function plotInputVsMetricCompare4(X, Y_Yu, Y_Liang, Y_ANN, Y_MINN, featShow, yLabelText, titleText, saveFolder, colors, markers, yLimits, fontName, outDPI, doExportFigure, shapContributionPercent)
    if nargin < 16
        shapContributionPercent = [];
    end

    if ~exist(saveFolder,'dir')
        mkdir(saveFolder);
    end

    modelKeys   = {'Yu','Liang','ANN','MINN'};
    modelLabels = {'Yu''s equation','Liang''s equation','ANN','MINN'};
    Ylist = {Y_Yu, Y_Liang, Y_ANN, Y_MINN};

    for j = 1:size(X,2)
        x = X(:,j);

        if isempty(yLimits)
            yAll = [];
            for m = 1:4
                yAll = [yAll; Ylist{m}(:,j)]; %#ok<AGROW>
            end
            yMin = min(yAll,[],'omitnan');
            yMax = max(yAll,[],'omitnan');
            yMargin = 0.08*(yMax - yMin);
            if ~isfinite(yMargin) || yMargin == 0
                yMargin = 1;
            end
            commonYLim = [yMin-yMargin, yMax+yMargin];
        else
            commonYLim = yLimits;
        end

        fig = figure('Color','w','Position',[80 80 1450 1000]);
        tiledlayout(2,2,'TileSpacing','loose','Padding','compact');

        for m = 1:4
            modelKey   = modelKeys{m};
            modelLabel = modelLabels{m};
            y = Ylist{m}(:,j);
            valid = ~isnan(x) & ~isnan(y);
            xx = x(valid);
            yy = y(valid);

            meanVal = mean(yy,'omitnan');
            maxVal  = max(yy,[],'omitnan');
            minVal  = min(yy,[],'omitnan');

            nexttile;
            hold on;
            box off;
            grid off;

            set(gca,...
    'FontName',fontName,...
    'FontSize',16,...
    'FontWeight','normal');

            ax = gca;

            c = colors.(modelKey);
            mk = markers.(modelKey);

            scatter(xx, yy, 120, mk, ...
                'MarkerFaceColor','none', ...
                'MarkerEdgeColor',c, ...
                'LineWidth',1.2);

            yline(0,'--k','LineWidth',1.0);
            yline(meanVal,'-',  'Color',c,'LineWidth',2.3);
            yline(maxVal, '--', 'Color',c,'LineWidth',1.7);
            yline(minVal, '--', 'Color',c,'LineWidth',1.7);

            xlabel(featShow{j}, 'FontName',fontName, 'FontSize',22);
            ylabel(yLabelText, 'FontName',fontName, 'FontSize',22);
            ylim(commonYLim);

            % =====================================================
            % Put Mean / Max / Min / Contribution below subplot title
            % - No text box inside plotting area
            % - Contribution is shown only for SHAP plots
            % =====================================================
            statLine = sprintf('Mean = %.3f   Max = %.3f   Min = %.3f', ...
                meanVal, maxVal, minVal);

            if ~isempty(shapContributionPercent)
                shapPct = shapContributionPercent(m,j);
                contributionLine = sprintf('Contribution = %.2f%%', shapPct);
                title({modelLabel; statLine; contributionLine}, ...
                    'FontName',fontName, ...
                    'FontSize',22, ...
                    'FontWeight','normal');
            else
                title({modelLabel; statLine}, ...
                    'FontName',fontName, ...
                    'FontSize',22, ...
                    'FontWeight','normal');
            end

% ให้เส้นแกนอยู่เฉพาะด้านล่างและซ้าย
            ax.XRuler.Axle.LineWidth = 1.2;
            ax.YRuler.Axle.LineWidth = 1.2;
        end

        sgtitle([titleText,' : ',featShow{j}], 'FontName',fontName, 'FontSize',22, 'FontWeight','bold');

        safeName = regexprep(yLabelText,'[^a-zA-Z0-9]','_');
        safeFeat = regexprep(featShow{j},'[^a-zA-Z0-9]','_');

        exportName = fullfile(saveFolder, ['Yu_Liang_ANN_MINN_Input_vs_', safeName, '_', safeFeat, '.png']);

        if doExportFigure
            exportgraphics(fig, exportName, 'Resolution', outDPI);
        end

        disp(['Saved: ', exportName]);
        close(fig);
    end
end

function plotRankCorrelationHeatmap(Tcorr, figTitle, outName, fontName, fontSize, outDPI, doExportFigure)
    models = {'Yu','Liang','ANN','MINN'};
    M = eye(numel(models));

    for r = 1:height(Tcorr)
        i = find(strcmp(models,Tcorr.Model_1{r}));
        j = find(strcmp(models,Tcorr.Model_2{r}));
        if ~isempty(i) && ~isempty(j)
            M(i,j) = Tcorr.Spearman_Rank_Correlation(r);
            M(j,i) = Tcorr.Spearman_Rank_Correlation(r);
        end
    end

    figure('Color','w');
    set(gcf,'Units','centimeters','Position',[5 5 14 12]);

    imagesc(M);
    axis equal tight;
    colorbar;
    caxis([-1 1]);

    xticks(1:numel(models));
    yticks(1:numel(models));
    xticklabels(models);
    yticklabels(models);

    set(gca, ...
    'FontName',fontName, ...
    'FontSize',16, ...
    'LineWidth',1.2, ...
    'TickLabelInterpreter','tex');

    ax = gca;


% แสดงเฉพาะแกนซ้ายและล่าง
    ax.Box = 'off';

% ให้เส้นแกนอยู่เฉพาะด้านล่างและซ้าย
     ax.XRuler.Axle.LineWidth = 1.2;
     ax.YRuler.Axle.LineWidth = 1.2;
    title(figTitle,'FontName',fontName,'FontSize',fontSize);

    for i = 1:numel(models)
        for j = 1:numel(models)
            text(j,i,sprintf('%.2f',M(i,j)), 'HorizontalAlignment','center', ...
                'FontName',fontName, 'FontSize',fontSize);
        end
    end

    if doExportFigure
        exportgraphics(gcf, outName, 'Resolution', outDPI);
    end
end