%% ============================================================
% ANN Grid Search - Data Loss Only + L2 Regularization
% Converted from PINN Grid Search by removing all physical losses
%
% Grid variables:
%   1) learnRate
%   2) numNeuron
%   3) numHL
%
% Main outputs:
%   - Excel: ANN_N_S_GridSearch_DataOnly_EarlyStopping_Results.xlsx
%   - MAT  : ANN_N_S_GridSearch_DataOnly_EarlyStopping_Results.mat
%   - Figures folder with surface/contour plots
%   - Best network saved as bestNet
%% ============================================================

clear; clc; close all;
tTotal = tic;

%% ============================================================
% 1) SETTINGS
%% ============================================================
trainFile = 'N_S_Training_DATASET_75.xlsx';
testFile  = 'N_S_Testing_DATASET_25.xlsx';

outFigDir = 'Figures_ANN_N_S_GridSearch';
if ~exist(outFigDir,'dir')
    mkdir(outFigDir);
end

excelFile = 'ANN_N_S_GridSearch_Results.xlsx';
matFile   = 'ANN_N_S_GridSearch_Results.mat';

%% ---------------- REPRODUCIBLE SETTING ----------------
fixedSeed = 1;
rng(fixedSeed,'twister');

% ใช้ CPU เพื่อให้ผลนิ่งที่สุด
try
    gpuDevice([]);
catch
end

%% ============================================================
% 2) GRID SEARCH RANGE
% ปรับช่วง grid ได้ตรงนี้
%% ============================================================
learningRates   = [1e-5 5e-5 1e-4 5e-4];
numNeurons      = [16 32 64 128];
numHiddenLayers = [2 3 4 5];

% numIter คือรอบสูงสุด (Maximum iteration) ของแต่ละ run
% ถ้า Total Loss เปลี่ยนแปลงน้อยกว่า lossTol ตามเงื่อนไขด้านล่าง จะหยุดก่อน numIter
numIter = 100000;

% ANN ใช้เฉพาะ data loss + L2 regularization
% ถ้าต้องการให้เหมือน ANN ปกติที่สุด ให้ lambda_data_total = 1
lambda_data_total = 0.005;
lambda_reg = 1e-7;

% เกณฑ์เลือก best: 'Testing_RMSE', 'Testing_MSE', 'Testing_R2', 'Training_RMSE', 'Training_MSE', 'Training_R2'
bestCriterion = 'Testing_RMSE';

% metric ที่ใช้ทำ surface plot: 'Testing_RMSE', 'Testing_MSE', 'Training_RMSE', 'Training_MSE'
plotMetricName = 'Testing_RMSE';

%% ---------------- EARLY STOPPING SETTING ----------------
% ตรวจทุก checkEvery iteration
% หยุดเมื่อ abs(TotalLoss_now - TotalLoss_before) < lossTol
lossTol      = 1e-5;
checkEvery   = 5000;
minIterCheck = 10000;

% ให้พิมพ์ผลตามรอบที่ตรวจ early stopping
printEvery = checkEvery;

totalRuns = length(learningRates) * length(numNeurons) * length(numHiddenLayers);
runID = 0;

%% ============================================================
% 3) LOAD DATA
%% ============================================================
Tr = readtable(trainFile,'VariableNamingRule','preserve');
Te = readtable(testFile,'VariableNamingRule','preserve');

Tr.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Tr.Properties.VariableNames));
Te.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Te.Properties.VariableNames));

X_tr = [Tr.D Tr.t Tr.fy Tr.fc Tr.L_D];
y_tr = Tr.N;

X_te = [Te.D Te.t Te.fy Te.f0c Te.L_D];
y_te = Te.N;

%% ============================================================
% 4) NORMALIZATION
%% ============================================================
muX  = mean(X_tr,1);
stdX = std(X_tr,0,1);
stdX(stdX==0) = 1;

Xn_tr = (X_tr - muX)./stdX;
Xn_te = (X_te - muX)./stdX;

muy  = mean(y_tr);
stdy = std(y_tr);
if stdy == 0
    stdy = 1;
end

yn_tr = (y_tr - muy)./stdy;

x_data = dlarray(Xn_tr','CB');
y_data = dlarray(yn_tr','CB');
x_te   = dlarray(Xn_te','CB');

Ntrain = size(Xn_tr,1);

%% ============================================================
% 5) STORAGE
%% ============================================================
nLR = length(learningRates);
nNN = length(numNeurons);
nHL = length(numHiddenLayers);

Run_grid = zeros(nLR,nNN,nHL);

Train_MSE_grid  = zeros(nLR,nNN,nHL);
Train_RMSE_grid = zeros(nLR,nNN,nHL);
Train_R2_grid   = zeros(nLR,nNN,nHL);
Train_MAE_grid  = zeros(nLR,nNN,nHL);
Train_A20_grid  = zeros(nLR,nNN,nHL);
Train_MeanRatio_grid = zeros(nLR,nNN,nHL);
Train_SDRatio_grid   = zeros(nLR,nNN,nHL);

Test_MSE_grid  = zeros(nLR,nNN,nHL);
Test_RMSE_grid = zeros(nLR,nNN,nHL);
Test_R2_grid   = zeros(nLR,nNN,nHL);
Test_MAE_grid  = zeros(nLR,nNN,nHL);
Test_A20_grid  = zeros(nLR,nNN,nHL);
Test_MeanRatio_grid = zeros(nLR,nNN,nHL);
Test_SDRatio_grid   = zeros(nLR,nNN,nHL);

Final_TotalLoss_grid = zeros(nLR,nNN,nHL);
Final_DataLoss_grid  = zeros(nLR,nNN,nHL);
Final_RegLoss_grid   = zeros(nLR,nNN,nHL);

StopIteration_grid = zeros(nLR,nNN,nHL);
Final_LossChange_grid = nan(nLR,nNN,nHL);
EarlyStopped_grid = false(nLR,nNN,nHL);

bestScore = inf;
bestNet = [];
bestInfo = struct();

%% ============================================================
% 6) GRID SEARCH LOOP
%% ============================================================
for h = 1:nHL

    nLayer = numHiddenLayers(h);

    for i = 1:nLR

        learnRate = learningRates(i);

        for j = 1:nNN

            runID = runID + 1;
            nNeuron = numNeurons(j);
            Run_grid(i,j,h) = runID;

            fprintf('\n====================================================\n');
            fprintf('Run %d/%d\n', runID, totalRuns);
            fprintf('Hidden Layers = %d | Neurons = %d | LR = %.1e | Max Iter = %d\n', ...
                nLayer, nNeuron, learnRate, numIter);
            fprintf('ANN loss: Data loss + L2 regularization only\n');
            fprintf('Early stopping: check every %d iter | lossTol = %.1e\n', checkEvery, lossTol);
            fprintf('====================================================\n');

            %% ----------------------------------------------------
            % CREATE NETWORK
            %% ----------------------------------------------------
            rng(fixedSeed,'twister');

            layers = [
                featureInputLayer(5,'Normalization','none','Name','input')
            ];

            for L = 1:nLayer
                layers = [
                    layers
                    fullyConnectedLayer(nNeuron,'Name',sprintf('fc%d',L))
                    tanhLayer('Name',sprintf('tanh%d',L))
                ];
            end

            layers = [
                layers
                fullyConnectedLayer(1,'Name','output')
            ];

            net = dlnetwork(layerGraph(layers));

            trailingAvg   = [];
            trailingAvgSq = [];

            %% ----------------------------------------------------
            % TRAINING LOOP WITH EARLY STOPPING
            %% ----------------------------------------------------
            lossHistoryRun = nan(numIter,1);
            stopIteration  = numIter;
            finalLossChange = NaN;
            earlyStopped = false;

            for it = 1:numIter

                [loss,grad,lossData,lossReg] = ...
                    dlfeval(@lossFun, net, x_data, y_data, ...
                    lambda_data_total, lambda_reg);

                [net,trailingAvg,trailingAvgSq] = adamupdate( ...
                    net, grad, trailingAvg, trailingAvgSq, it, learnRate);

                currentTotalLoss = gather(extractdata(loss));
                lossHistoryRun(it) = currentTotalLoss;

                if mod(it,printEvery)==0 || it==1 || it==numIter
                    fprintf('Run %d | Iter %5d | Total %.3e | Data %.3e | Reg %.3e\n', ...
                        runID, it, currentTotalLoss, ...
                        gather(extractdata(lossData)), gather(extractdata(lossReg)));
                end

                %% ============================================
                % EARLY STOPPING BASED ON TOTAL LOSS CHANGE
                % ตรวจทุก checkEvery iteration
                % ถ้า abs(loss_now - loss_before) < lossTol ให้หยุด run นี้
                %% ============================================
                if mod(it,checkEvery)==0 && it >= minIterCheck

                    finalLossChange = abs(lossHistoryRun(it) - lossHistoryRun(it-checkEvery));

                    fprintf('Run %d | Loss Change over last %d iter = %.6e\n', ...
                        runID, checkEvery, finalLossChange);

                    if finalLossChange < lossTol
                        stopIteration = it;
                        earlyStopped = true;

                        fprintf('Run %d | EARLY STOP at Iter %d | Loss Change %.6e < %.6e\n', ...
                            runID, stopIteration, finalLossChange, lossTol);
                        break;
                    end
                end
            end

            StopIteration_grid(i,j,h) = stopIteration;
            Final_LossChange_grid(i,j,h) = finalLossChange;
            EarlyStopped_grid(i,j,h) = earlyStopped;

            Final_TotalLoss_grid(i,j,h) = gather(extractdata(loss));
            Final_DataLoss_grid(i,j,h)  = gather(extractdata(lossData));
            Final_RegLoss_grid(i,j,h)   = gather(extractdata(lossReg));

            %% ----------------------------------------------------
            % TRAINING PREDICTION
            %% ----------------------------------------------------
            y_pred_tr = forward(net,x_data,'Outputs','output');
            y_pred_tr = gather(extractdata(y_pred_tr))';
            y_pred_tr = y_pred_tr .* stdy + muy;

            [MAE_tr,MSE_tr,RMSE_tr,R2_tr,A20_tr,MeanRatio_tr,SDRatio_tr] = calc_metrics(y_tr,y_pred_tr);

            Train_MSE_grid(i,j,h)  = MSE_tr;
            Train_RMSE_grid(i,j,h) = RMSE_tr;
            Train_R2_grid(i,j,h)   = R2_tr;
            Train_MAE_grid(i,j,h)  = MAE_tr;
            Train_A20_grid(i,j,h)  = A20_tr;
            Train_MeanRatio_grid(i,j,h) = MeanRatio_tr;
            Train_SDRatio_grid(i,j,h)   = SDRatio_tr;

            %% ----------------------------------------------------
            % TESTING PREDICTION
            %% ----------------------------------------------------
            y_pred_te = forward(net,x_te,'Outputs','output');
            y_pred_te = gather(extractdata(y_pred_te))';
            y_pred_te = y_pred_te .* stdy + muy;

            [MAE_te,MSE_te,RMSE_te,R2_te,A20_te,MeanRatio_te,SDRatio_te] = calc_metrics(y_te,y_pred_te);

            Test_MSE_grid(i,j,h)  = MSE_te;
            Test_RMSE_grid(i,j,h) = RMSE_te;
            Test_R2_grid(i,j,h)   = R2_te;
            Test_MAE_grid(i,j,h)  = MAE_te;
            Test_A20_grid(i,j,h)  = A20_te;
            Test_MeanRatio_grid(i,j,h) = MeanRatio_te;
            Test_SDRatio_grid(i,j,h)   = SDRatio_te;

            %% ----------------------------------------------------
            % BEST MODEL SELECTION
            %% ----------------------------------------------------
            score = get_score(bestCriterion, ...
                MSE_tr, RMSE_tr, R2_tr, MSE_te, RMSE_te, R2_te);

            if score < bestScore
                bestScore = score;
                bestNet = net;

                bestInfo.Run = runID;
                bestInfo.HiddenLayers = nLayer;
                bestInfo.NumNeurons = nNeuron;
                bestInfo.LearningRate = learnRate;
                bestInfo.BestCriterion = bestCriterion;
                bestInfo.BestScore = bestScore;
                bestInfo.StopIteration = stopIteration;
                bestInfo.EarlyStopped = earlyStopped;
                bestInfo.FinalLossChange = finalLossChange;

                bestInfo.Training_MAE = MAE_tr;
                bestInfo.Training_MSE = MSE_tr;
                bestInfo.Training_RMSE = RMSE_tr;
                bestInfo.Training_R2 = R2_tr;
                bestInfo.Training_A20 = A20_tr;
                bestInfo.Training_MeanRatio = MeanRatio_tr;
                bestInfo.Training_SDRatio = SDRatio_tr;

                bestInfo.Testing_MAE = MAE_te;
                bestInfo.Testing_MSE = MSE_te;
                bestInfo.Testing_RMSE = RMSE_te;
                bestInfo.Testing_R2 = R2_te;
                bestInfo.Testing_A20 = A20_te;
                bestInfo.Testing_MeanRatio = MeanRatio_te;
                bestInfo.Testing_SDRatio = SDRatio_te;
            end

            fprintf('Run %d Finished | StopIteration = %d | EarlyStopped = %d | FinalLossChange = %.6e\n', ...
                runID, stopIteration, earlyStopped, finalLossChange);
            fprintf('Train: RMSE %.3f | R2 %.5f | A20 %.2f %% | Mean %.4f\n', RMSE_tr, R2_tr, A20_tr, MeanRatio_tr);
            fprintf('Test : RMSE %.3f | R2 %.5f | A20 %.2f %% | Mean %.4f\n', RMSE_te, R2_te, A20_te, MeanRatio_te);
            fprintf('Current Best: Run %d | HL %d | Neuron %d | LR %.1e | %s score %.5g\n', ...
                bestInfo.Run, bestInfo.HiddenLayers, bestInfo.NumNeurons, ...
                bestInfo.LearningRate, bestCriterion, bestInfo.BestScore);

        end
    end
end

%% ============================================================
% 7) PRINT BEST RESULT
%% ============================================================
fprintf('\n================ BEST GRID SEARCH RESULT ================\n');
fprintf('Best Criterion       = %s\n', bestCriterion);
fprintf('Best Run             = %d\n', bestInfo.Run);
fprintf('Best Hidden Layers   = %d\n', bestInfo.HiddenLayers);
fprintf('Best Neurons         = %d\n', bestInfo.NumNeurons);
fprintf('Best Learning Rate   = %.1e\n', bestInfo.LearningRate);
fprintf('Best Score           = %.6g\n', bestInfo.BestScore);
fprintf('Stop Iteration        = %d\n', bestInfo.StopIteration);
fprintf('Early Stopped         = %d\n', bestInfo.EarlyStopped);
fprintf('Final Loss Change     = %.6e\n', bestInfo.FinalLossChange);
fprintf('Training: MAE %.3f | MSE %.3f | RMSE %.3f | R2 %.5f | A20 %.2f %% | Mean %.4f | SD %.4f\n', ...
    bestInfo.Training_MAE, bestInfo.Training_MSE, bestInfo.Training_RMSE, bestInfo.Training_R2, ...
    bestInfo.Training_A20, bestInfo.Training_MeanRatio, bestInfo.Training_SDRatio);
fprintf('Testing : MAE %.3f | MSE %.3f | RMSE %.3f | R2 %.5f | A20 %.2f %% | Mean %.4f | SD %.4f\n', ...
    bestInfo.Testing_MAE, bestInfo.Testing_MSE, bestInfo.Testing_RMSE, bestInfo.Testing_R2, ...
    bestInfo.Testing_A20, bestInfo.Testing_MeanRatio, bestInfo.Testing_SDRatio);
fprintf('==========================================================\n');

%% ============================================================
% 8) SURFACE + CONTOUR PLOT FOR EACH HIDDEN LAYER
%% ============================================================
PlotGrid = get_plot_grid(plotMetricName, ...
    Train_MSE_grid, Train_RMSE_grid, Test_MSE_grid, Test_RMSE_grid);

[Xgrid, Ygrid] = meshgrid(1:length(numNeurons), 1:length(learningRates));

for h = 1:length(numHiddenLayers)

    nLayer = numHiddenLayers(h);
    Z = PlotGrid(:,:,h);

    fig = figure;
    set(fig,'Position',[100 100 950 720]);

    surf(Xgrid,Ygrid,Z, ...
        'EdgeColor','k', ...
        'LineWidth',0.45, ...
        'FaceColor','interp');

    hold on;

    zmin = min(Z(:));
    zmax = max(Z(:));

    if zmax > zmin
        zOffset = zmin - 0.05*(zmax-zmin);
    else
        zOffset = zmin - 0.05*max(abs(zmin),1);
    end

    contour3(Xgrid,Ygrid,Z,20,'LineWidth',0.8);

    hContour = findobj(gca,'Type','Contour');
    for c = 1:length(hContour)
        hContour(c).ZData = zOffset .* ones(size(hContour(c).ZData));
    end

    [bestMetric_layer,idxLayer] = min(Z(:));
    [iLayer,jLayer] = ind2sub(size(Z),idxLayer);

    bestRun_layer    = Run_grid(iLayer,jLayer,h);
    bestLR_layer     = learningRates(iLayer);
    bestNeuron_layer = numNeurons(jLayer);

    plot3(jLayer,iLayer,bestMetric_layer,'ro', ...
        'MarkerSize',10,'MarkerFaceColor','r','LineWidth',1.5);

    title(sprintf([ ...
        'ANN Grid Search Surface (%s)\n' ...
        'Hidden Layers = %d | Min = %.3f | Run = %d | LR = %.1e | Neuron = %d'], ...
        strrep(plotMetricName,'_',' '), nLayer, bestMetric_layer, ...
        bestRun_layer, bestLR_layer, bestNeuron_layer), ...
        'FontName','Times New Roman','FontSize',15,'FontWeight','bold');

    xlabel('Number of neurons','FontName','Times New Roman','FontSize',14);
    ylabel('Learning rate','FontName','Times New Roman','FontSize',14);
    zlabel(strrep(plotMetricName,'_',' '),'FontName','Times New Roman','FontSize',14);

    xticks(1:length(numNeurons));
    yticks(1:length(learningRates));
    xticklabels(string(numNeurons));
    yticklabels(compose('%.0e',learningRates));

    xlim([1 length(numNeurons)]);
    ylim([1 length(learningRates)]);

    colormap(turbo);
    cb = colorbar;
    cb.Label.String = strrep(plotMetricName,'_',' ');
    cb.Label.FontName = 'Times New Roman';
    cb.Label.FontSize = 13;

    grid on; box on;
    set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1.2);
    zlim([zOffset zmax]);
    view(-135,30);
    camlight headlight;
    lighting gouraud;
    hold off;

    exportgraphics(fig, fullfile(outFigDir, ...
        sprintf('ANN_%s_Surface_Contour_HiddenLayer_%d.png', plotMetricName, nLayer)), ...
        'Resolution',1200);

    exportgraphics(fig, fullfile(outFigDir, ...
        sprintf('ANN_%s_Surface_Contour_HiddenLayer_%d.tif', plotMetricName, nLayer)), ...
        'Resolution',1200);
end

%% ============================================================
% 9) COMBINED SURFACE PLOT
%% ============================================================
figAll = figure;
set(figAll,'Position',[100 100 1500 1100]);

tiledlayout(2,ceil(length(numHiddenLayers)/2), ...
    'TileSpacing','compact','Padding','compact');

for h = 1:length(numHiddenLayers)

    nexttile;

    nLayer = numHiddenLayers(h);
    Z = PlotGrid(:,:,h);

    surf(Xgrid,Ygrid,Z, ...
        'EdgeColor','k', ...
        'LineWidth',0.35, ...
        'FaceColor','interp');

    hold on;

    zmin = min(Z(:));
    zmax = max(Z(:));

    if zmax > zmin
        zOffset = zmin - 0.05*(zmax-zmin);
    else
        zOffset = zmin - 0.05*max(abs(zmin),1);
    end

    contour3(Xgrid,Ygrid,Z,15,'LineWidth',0.6);

    hContour = findobj(gca,'Type','Contour');
    for c = 1:length(hContour)
        hContour(c).ZData = zOffset .* ones(size(hContour(c).ZData));
    end

    [bestMetric_layer,idxLayer] = min(Z(:));
    [iLayer,jLayer] = ind2sub(size(Z),idxLayer);
    bestRun_layer = Run_grid(iLayer,jLayer,h);

    plot3(jLayer,iLayer,bestMetric_layer,'ro', ...
        'MarkerSize',8,'MarkerFaceColor','r','LineWidth',1.3);

    xlabel('Number of neurons');
    ylabel('Learning rate');
    zlabel(strrep(plotMetricName,'_',' '));

    title(sprintf('(%c) HL = %d | Min = %.2f | Run = %d', ...
        char(96+h), nLayer, bestMetric_layer, bestRun_layer), ...
        'FontName','Times New Roman','FontSize',12,'FontWeight','bold');

    xticks(1:length(numNeurons));
    yticks(1:length(learningRates));
    xticklabels(string(numNeurons));
    yticklabels(compose('%.0e',learningRates));

    xlim([1 length(numNeurons)]);
    ylim([1 length(learningRates)]);

    colormap(turbo);
    grid on; box on;
    zlim([zOffset zmax]);
    view(-135,30);

    set(gca,'FontName','Times New Roman','FontSize',11,'LineWidth',1.0);
    hold off;
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = strrep(plotMetricName,'_',' ');
cb.Label.FontName = 'Times New Roman';
cb.Label.FontSize = 13;

exportgraphics(figAll, fullfile(outFigDir, ...
    sprintf('ANN_%s_Surface_Contour_All_HiddenLayers.png', plotMetricName)), ...
    'Resolution',1000);

exportgraphics(figAll, fullfile(outFigDir, ...
    sprintf('ANN_%s_Surface_Contour_All_HiddenLayers.tif', plotMetricName)), ...
    'Resolution',1000);

%% ============================================================
% 10) EXPORT RESULT TO EXCEL
%% ============================================================
ResultTable = table;
row = 1;

for h = 1:length(numHiddenLayers)
    for i = 1:length(learningRates)
        for j = 1:length(numNeurons)

            ResultTable.Run(row,1)          = Run_grid(i,j,h);
            ResultTable.HiddenLayers(row,1) = numHiddenLayers(h);
            ResultTable.LearningRate(row,1) = learningRates(i);
            ResultTable.NumNeurons(row,1)   = numNeurons(j);
            ResultTable.StopIteration(row,1) = StopIteration_grid(i,j,h);
            ResultTable.EarlyStopped(row,1) = EarlyStopped_grid(i,j,h);
            ResultTable.FinalLossChange_CheckEvery(row,1) = Final_LossChange_grid(i,j,h);

            ResultTable.Training_MAE(row,1)  = Train_MAE_grid(i,j,h);
            ResultTable.Training_MSE(row,1)  = Train_MSE_grid(i,j,h);
            ResultTable.Training_RMSE(row,1) = Train_RMSE_grid(i,j,h);
            ResultTable.Training_R2(row,1)   = Train_R2_grid(i,j,h);
            ResultTable.Training_A20(row,1)  = Train_A20_grid(i,j,h);
            ResultTable.Training_Mean_Npred_Nexp(row,1) = Train_MeanRatio_grid(i,j,h);
            ResultTable.Training_SD_Npred_Nexp(row,1)   = Train_SDRatio_grid(i,j,h);

            ResultTable.Testing_MAE(row,1)  = Test_MAE_grid(i,j,h);
            ResultTable.Testing_MSE(row,1)  = Test_MSE_grid(i,j,h);
            ResultTable.Testing_RMSE(row,1) = Test_RMSE_grid(i,j,h);
            ResultTable.Testing_R2(row,1)   = Test_R2_grid(i,j,h);
            ResultTable.Testing_A20(row,1)  = Test_A20_grid(i,j,h);
            ResultTable.Testing_Mean_Npred_Nexp(row,1) = Test_MeanRatio_grid(i,j,h);
            ResultTable.Testing_SD_Npred_Nexp(row,1)   = Test_SDRatio_grid(i,j,h);

            ResultTable.Final_TotalLoss(row,1) = Final_TotalLoss_grid(i,j,h);
            ResultTable.Final_DataLoss(row,1)  = Final_DataLoss_grid(i,j,h);
            ResultTable.Final_RegLoss(row,1)   = Final_RegLoss_grid(i,j,h);

            row = row + 1;
        end
    end
end

writetable(ResultTable,excelFile,'Sheet','GRID_SEARCH_RESULTS');

BestResult = struct2table(bestInfo);
writetable(BestResult,excelFile,'Sheet','BEST_RESULT');

LossWeights = table( ...
    ["lambda_data_total"; "lambda_reg"; "lossTol"; "checkEvery"; "minIterCheck"], ...
    [lambda_data_total; lambda_reg; lossTol; checkEvery; minIterCheck], ...
    'VariableNames', {'Name','Value'} );
writetable(LossWeights,excelFile,'Sheet','LOSS_WEIGHTS');

GridSetting = table( ...
    fixedSeed, numIter, lossTol, checkEvery, minIterCheck, string(bestCriterion), string(plotMetricName), Ntrain, ...
    'VariableNames', {'FixedSeed','MaxNumIter','LossTol','CheckEvery','MinIterCheck','BestCriterion','PlotMetric','Ntrain'} );
writetable(GridSetting,excelFile,'Sheet','GRID_SETTINGS');

for h = 1:length(numHiddenLayers)

    nLayer = numHiddenLayers(h);

    Train_RMSE_Table = array2table(Train_RMSE_grid(:,:,h), ...
        'VariableNames', compose('Neuron_%d',numNeurons));
    Train_RMSE_Table.LearningRate = learningRates';
    Train_RMSE_Table = movevars(Train_RMSE_Table,'LearningRate','Before',1);
    writetable(Train_RMSE_Table,excelFile,'Sheet',sprintf('Train_RMSE_HL_%d',nLayer));

    Test_RMSE_Table = array2table(Test_RMSE_grid(:,:,h), ...
        'VariableNames', compose('Neuron_%d',numNeurons));
    Test_RMSE_Table.LearningRate = learningRates';
    Test_RMSE_Table = movevars(Test_RMSE_Table,'LearningRate','Before',1);
    writetable(Test_RMSE_Table,excelFile,'Sheet',sprintf('Test_RMSE_HL_%d',nLayer));

    Train_MSE_Table = array2table(Train_MSE_grid(:,:,h), ...
        'VariableNames', compose('Neuron_%d',numNeurons));
    Train_MSE_Table.LearningRate = learningRates';
    Train_MSE_Table = movevars(Train_MSE_Table,'LearningRate','Before',1);
    writetable(Train_MSE_Table,excelFile,'Sheet',sprintf('Train_MSE_HL_%d',nLayer));

    Test_MSE_Table = array2table(Test_MSE_grid(:,:,h), ...
        'VariableNames', compose('Neuron_%d',numNeurons));
    Test_MSE_Table.LearningRate = learningRates';
    Test_MSE_Table = movevars(Test_MSE_Table,'LearningRate','Before',1);
    writetable(Test_MSE_Table,excelFile,'Sheet',sprintf('Test_MSE_HL_%d',nLayer));
end

fprintf('\nGrid search results exported to: %s\n',excelFile);
fprintf('Figures saved in folder: %s\n',outFigDir);

%% ============================================================
% 11) SAVE MAT FILE
%% ============================================================
save(matFile, ...
    'bestNet','bestInfo','bestCriterion','plotMetricName', ...
    'fixedSeed','learningRates','numNeurons','numHiddenLayers','numIter', ...
    'lossTol','checkEvery','minIterCheck','StopIteration_grid','Final_LossChange_grid','EarlyStopped_grid', ...
    'Run_grid', ...
    'Train_MSE_grid','Train_RMSE_grid','Train_R2_grid','Train_MAE_grid','Train_A20_grid', ...
    'Train_MeanRatio_grid','Train_SDRatio_grid', ...
    'Test_MSE_grid','Test_RMSE_grid','Test_R2_grid','Test_MAE_grid','Test_A20_grid', ...
    'Test_MeanRatio_grid','Test_SDRatio_grid', ...
    'Final_TotalLoss_grid','Final_DataLoss_grid','Final_RegLoss_grid', ...
    'muX','stdX','muy','stdy', ...
    'lambda_data_total','lambda_reg','Ntrain');

fprintf('MAT file saved: %s\n',matFile);
fprintf('Best network saved inside MAT file as variable: bestNet\n');
fprintf('Total time = %.2f sec\n',toc(tTotal));

%% ============================================================
% LOCAL HELPER: LOSS FUNCTION
% ANN version:
%   Total loss = lambda_data_total * MSE(normalized prediction, normalized target)
%                + lambda_reg * L2(parameters)
%
% ตัด physical loss ออกทั้งหมด:
%   - ไม่มี Yu loss
%   - ไม่มี Liang loss
%   - ไม่มี L/D constraint loss
%% ============================================================
function [loss,grad,lossData,lossReg] = lossFun(net,x,y,lambda_data_total,lambda_reg)

    %% DATA LOSS
    yp_data = forward(net,x,'Outputs','output');
    lossData = mean((yp_data - y).^2,'all');

    %% L2 REGULARIZATION
    reg = dlarray(0);
    vals = net.Learnables.Value;

    for i = 1:numel(vals)
        reg = reg + sum(vals{i}.^2,'all');
    end

    lossReg = lambda_reg .* reg;

    %% TOTAL LOSS
    loss = lambda_data_total .* lossData + lossReg;

    grad = dlgradient(loss,net.Learnables);

end

%% ============================================================
% LOCAL HELPER: METRICS
%% ============================================================
function [MAE,MSE,RMSE,R2,A20,MeanRatio,SDRatio] = calc_metrics(y_true,y_pred)

    MAE  = mean(abs(y_pred - y_true));
    MSE  = mean((y_pred - y_true).^2);
    RMSE = sqrt(MSE);

    denom = sum((y_true - mean(y_true)).^2);
    if denom == 0
        R2 = NaN;
    else
        R2 = 1 - sum((y_pred - y_true).^2) / denom;
    end

    ratio = y_pred ./ y_true;
    valid = isfinite(ratio);

    if any(valid)
        MeanRatio = mean(ratio(valid));
        SDRatio   = std(ratio(valid));
        A20 = mean((ratio(valid) >= 0.8) & (ratio(valid) <= 1.2)) * 100;
    else
        MeanRatio = NaN;
        SDRatio   = NaN;
        A20 = NaN;
    end
end

%% ============================================================
% LOCAL HELPER: SCORE FOR BEST MODEL
%% ============================================================
function score = get_score(bestCriterion, MSE_tr, RMSE_tr, R2_tr, MSE_te, RMSE_te, R2_te)

    switch string(bestCriterion)
        case "Testing_RMSE"
            score = RMSE_te;
        case "Testing_MSE"
            score = MSE_te;
        case "Testing_R2"
            score = -R2_te;
        case "Training_RMSE"
            score = RMSE_tr;
        case "Training_MSE"
            score = MSE_tr;
        case "Training_R2"
            score = -R2_tr;
        otherwise
            error('Unknown bestCriterion: %s', bestCriterion);
    end
end

%% ============================================================
% LOCAL HELPER: SELECT GRID FOR PLOT
%% ============================================================
function PlotGrid = get_plot_grid(plotMetricName, Train_MSE_grid, Train_RMSE_grid, Test_MSE_grid, Test_RMSE_grid)

    switch string(plotMetricName)
        case "Training_MSE"
            PlotGrid = Train_MSE_grid;
        case "Training_RMSE"
            PlotGrid = Train_RMSE_grid;
        case "Testing_MSE"
            PlotGrid = Test_MSE_grid;
        case "Testing_RMSE"
            PlotGrid = Test_RMSE_grid;
        otherwise
            error('Unknown plotMetricName: %s', plotMetricName);
    end
end
