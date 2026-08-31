%% ============================================================
% Calculate Prediction Accuracy using Ding, Liang, and Yu formulas
% For Short CCFST Columns under Concentric Loading
% ============================================================
clear; clc; close all;

%% ===================== FILES =====================
trainFile = 'N_S_Training_DATASET_75.xlsx';
testFile  = 'N_S_Testing_DATASET_External.xlsx';

outExcel = 'Formula_Ding_Liang_Yu_Accuracy.xlsx';

%% ===================== READ DATA =====================
Tr = readtable(trainFile,'VariableNamingRule','preserve');
Te = readtable(testFile,'VariableNamingRule','preserve');

Tr.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Tr.Properties.VariableNames));
Te.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Te.Properties.VariableNames));

%% ===================== CALCULATE FORMULAS =====================
TrainResult = calculateAllFormulas(Tr);
TestResult  = calculateAllFormulas(Te);

%% ===================== CALCULATE METRICS =====================
MetricsTrain = calculateMetricsTable(TrainResult, "Training");
MetricsTest  = calculateMetricsTable(TestResult, "Testing");

Metrics = [MetricsTrain; MetricsTest];

%% ===================== VALID SUMMARY =====================
ValidSummaryTrain = calculateValidSummary(TrainResult, "Training");
ValidSummaryTest  = calculateValidSummary(TestResult, "Testing");

ValidSummary = [ValidSummaryTrain; ValidSummaryTest];

%% ===================== EXPORT =====================
writetable(TrainResult,outExcel,'Sheet','TRAINING_PREDICTION');
writetable(TestResult,outExcel,'Sheet','TESTING_PREDICTION');
writetable(Metrics,outExcel,'Sheet','METRICS');
writetable(ValidSummary,outExcel,'Sheet','VALID_DATA_SUMMARY');

disp('====================================================')
disp('Export completed:')
disp(outExcel)
disp('====================================================')

%% ===================== PRINT VALID SUMMARY =====================
disp(' ')
disp('================ VALID DATA SUMMARY ================')
disp(ValidSummary)
disp('=====================================================')

%% ===================== PRINT METRICS =====================
disp(' ')
disp('================ ACCURACY METRICS ================')

for i = 1:height(Metrics)

    fprintf('\nDataset : %s\n', Metrics.Dataset{i});
    fprintf('Method  : %s\n', Metrics.Method(i));

    fprintf('Valid Data        = %d\n', Metrics.ValidData(i));

    fprintf('R2                = %.4f\n', Metrics.R2(i));
    fprintf('MAE               = %.3f kN\n', Metrics.MAE(i));
    fprintf('MSE               = %.3f\n', Metrics.MSE(i));
    fprintf('RMSE              = %.3f kN\n', Metrics.RMSE(i));

    fprintf('A20-index         = %.2f %%\n', ...
        Metrics.A20_index_percent(i));

    fprintf('Mean(Npred/Nexp)  = %.4f\n', ...
        Metrics.Mean_Npred_Nexp(i));

    fprintf('SD(Npred/Nexp)    = %.4f\n', ...
        Metrics.SD_Npred_Nexp(i));

    fprintf('COV(Npred/Nexp)   = %.2f %%\n', ...
        Metrics.COV_Npred_Nexp_percent(i));

    fprintf('--------------------------------------------------\n')

end

disp('==================================================')

%% ============================================================
% FUNCTION: Calculate Ding, Liang, Yu
% ============================================================
function Result = calculateAllFormulas(T)

    D  = T.D;
    t  = T.t;
    Fy = T.fy;
    Fc = T.fc;
    N_exp = T.N;

    if ismember('L_D', T.Properties.VariableNames)
        L_D = T.L_D;
    elseif ismember('L_D_', T.Properties.VariableNames)
        L_D = T.L_D_;
    elseif ismember('L_D__', T.Properties.VariableNames)
        L_D = T.L_D__;
    else
        L_D = nan(size(D));
    end

    Di = max(D - 2.*t, 0);

    As = pi/4 .* (D.^2 - Di.^2);
    Ac = pi/4 .* Di.^2;

    %% ===================== Ding Formula =====================
    K = nan(size(Fy));

    idx_CS_CC = Fy < 500  & Fc < 100;
    idx_HS_CC = Fy >= 500 & Fc < 100;
    idx_CS_HC = Fy < 500  & Fc >= 100;
    idx_HS_HC = Fy >= 500 & Fc >= 100;

    K(idx_CS_CC) = 1.62;
    K(idx_HS_CC) = 1.52;
    K(idx_CS_HC) = 1.44;
    K(idx_HS_HC) = 1.48;

    validDing = Fc > 0 & Fy > 0 & Ac > 0 & As > 0 & ~isnan(K);

    N_Ding = nan(size(D));

    N_Ding(validDing) = ...
        (Fc(validDing).*Ac(validDing) + ...
        K(validDing).*Fy(validDing).*As(validDing)) ./ 1000;

    %% ===================== Yu Unified Formula =====================
    N_Yu = nan(size(D));

    validYu =                  Ac > 0 & As > 0;

    if any(validYu)

        Dv  = D(validYu);
        tv  = t(validYu);
        fyv = Fy(validYu);
        fcv = Fc(validYu);

        Div = max(Dv - 2.*tv, 0);

        Asv = pi/4 .* (Dv.^2 - Div.^2);
        Acv = pi/4 .* Div.^2;

        fck = 0.8 .* fcv;

        Ascv = Asv + Acv;

        beta = Asv ./ Ascv;

        Omega = ones(size(Acv));

        xi_sc = (Asv .* fyv) ./ (Acv .* fck);

        etaYu = (Omega .* xi_sc) ./ ...
            ( ...
            (2.0.*Omega + 0.05.*xi_sc + ...
            (0.2.*fck./fyv - 0.05).*xi_sc.*Omega) ...
            .* (Omega + xi_sc) ...
            );

        fsc = (1 + etaYu) .* ((1 - beta).*fck + beta.*fyv);

        N_Yu(validYu) = fsc .* Ascv ./ 1000;

    end

    %% ===================== Liang & Fragomeni Formula =====================
    Dt = D ./ t;
    ratio_fc_fy_all = Fc ./ Fy;

    N_Liang = nan(size(D));

    validLiang = ...
        (Dt > 0) & ...
        (Dt <= 150) & ...
        (Fc > 0) & ...
        (Fy > 0) & ...
        (t > 0) & ...
        (Di > 0) & ...
        (ratio_fc_fy_all >= 0.04) & ...
        (ratio_fc_fy_all <= 0.20);

    if any(validLiang)

        Dv  = D(validLiang);
        tv  = t(validLiang);
        fyv = Fy(validLiang);
        fcv = Fc(validLiang);
        Div = Di(validLiang);

        Asv = As(validLiang);
        Acv = Ac(validLiang);

        Dtv = Dv ./ tv;
        Dcv = Div;

        gamma_c = 1.85 .* Dcv.^(-0.135);
        gamma_c = min(max(gamma_c,0.85),1.0);

        gamma_s = 1.458 .* Dtv.^(-0.1);
        gamma_s = min(max(gamma_s,0.9),1.1);

        frp = zeros(size(Dtv));

        idx1 = Dtv <= 47;
        idx2 = Dtv > 47 & Dtv <= 150;

        if any(idx1)

            Dt1 = Dtv(idx1);
            fc1 = fcv(idx1);
            fy1 = fyv(idx1);
            D1  = Dv(idx1);
            t1  = tv(idx1);

            ratio1 = fc1 ./ fy1;

            ve0 = 0.881e-6 .* Dt1.^3 ...
                - 2.58e-4 .* Dt1.^2 ...
                + 1.953e-2 .* Dt1 ...
                + 0.4011;

            ve = 0.2312 ...
                + 0.3582 .* ve0 ...
                - 0.1524 .* ratio1 ...
                + 4.843  .* ve0 .* ratio1 ...
                - 9.169  .* ratio1.^2;

            vs = 0.5;

            frp(idx1) = 0.7 .* (ve - vs) .* ...
                        (2.*t1 ./ (D1 - 2.*t1)) .* fy1;

            frp(idx1) = max(frp(idx1),0);

        end

        if any(idx2)

            frp(idx2) = ...
                (0.006241 - 0.0000357 .* Dtv(idx2)) .* fyv(idx2);

            frp(idx2) = max(frp(idx2),0);

        end

        Nu_Liang_valid = ...
            ((gamma_c .* fcv + 4.1 .* frp) .* Acv + ...
            gamma_s .* fyv .* Asv) ./ 1000;

        N_Liang(validLiang) = Nu_Liang_valid;

    end

    %% ===================== RESULT TABLE =====================
    Result = table( ...
        D, t, Fy, Fc, L_D, N_exp, ...
        N_Ding, N_Yu, N_Liang, ...
        N_Ding - N_exp, ...
        N_Yu - N_exp, ...
        N_Liang - N_exp, ...
        N_Ding ./ N_exp, ...
        N_Yu ./ N_exp, ...
        N_Liang ./ N_exp, ...
        'VariableNames', { ...
        'D','t','Fy','Fc','L_D','N_exp', ...
        'N_Ding','N_Yu','N_Liang', ...
        'Error_Ding','Error_Yu','Error_Liang', ...
        'Ratio_Ding','Ratio_Yu','Ratio_Liang'} );

end

%% ============================================================
% FUNCTION: Accuracy Metrics
% ============================================================
function Metrics = calculateMetricsTable(Result, datasetName)

    methods = ["Ding"; "Yu"; "Liang"];

    MAE = zeros(3,1);
    MSE = zeros(3,1);
    RMSE = zeros(3,1);
    R2 = zeros(3,1);
    A20 = zeros(3,1);
    MeanRatio = zeros(3,1);
    SDRatio = zeros(3,1);
    COVRatio = zeros(3,1);
    ValidData = zeros(3,1);

    N_exp = Result.N_exp;

    for i = 1:3

        method = methods(i);
        predName = "N_" + method;

        N_pred = Result.(predName);

        valid = ~isnan(N_pred) & ...
                ~isnan(N_exp) & ...
                N_exp > 0;

        y = N_exp(valid);
        yp = N_pred(valid);

        ratio = yp ./ y;

        MAE(i) = mean(abs(yp - y));
        MSE(i) = mean((yp - y).^2);
        RMSE(i) = sqrt(MSE(i));

        R2(i) = 1 - sum((yp - y).^2) / ...
                    sum((y - mean(y)).^2);

        A20(i) = mean(ratio >= 0.8 & ratio <= 1.2) * 100;

        MeanRatio(i) = mean(ratio);
        SDRatio(i) = std(ratio);
        COVRatio(i) = SDRatio(i) ./ MeanRatio(i) .* 100;

        ValidData(i) = sum(valid);

    end

    Metrics = table( ...
        repmat(datasetName,3,1), ...
        methods, ...
        ValidData, ...
        R2, MAE, MSE, RMSE, A20, ...
        MeanRatio, SDRatio, COVRatio, ...
        'VariableNames', { ...
        'Dataset','Method','ValidData', ...
        'R2','MAE','MSE','RMSE','A20_index_percent', ...
        'Mean_Npred_Nexp','SD_Npred_Nexp', ...
        'COV_Npred_Nexp_percent'} );

end

%% ============================================================
% FUNCTION: Valid / Invalid Summary
% ============================================================
function Summary = calculateValidSummary(Result, datasetName)

    methods = ["Ding"; "Yu"; "Liang"];

    ValidData = zeros(3,1);
    InvalidData = zeros(3,1);
    TotalData = zeros(3,1);

    totalN = height(Result);

    for i = 1:3

        method = methods(i);
        predName = "N_" + method;

        pred = Result.(predName);

        valid = ~isnan(pred);

        ValidData(i) = sum(valid);
        InvalidData(i) = sum(~valid);
        TotalData(i) = totalN;

    end

    Summary = table( ...
        repmat(datasetName,3,1), ...
        methods, ...
        TotalData, ...
        ValidData, ...
        InvalidData, ...
        'VariableNames', { ...
        'Dataset', ...
        'Method', ...
        'TotalData', ...
        'ValidData', ...
        'InvalidData'} );

end