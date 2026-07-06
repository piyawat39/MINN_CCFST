clear; clc; close all;
tTotal = tic;

%% ============================================================
% 1) SETTINGS
% ============================================================
showLivePlot = true;
saveVideo    = true;

trainFile = 'N_S_Training_DATASET_75.xlsx';
testFile  = 'N_S_Testing_DATASET_25.xlsx';

%% ---------------- REPRODUCIBLE SETTING ----------------
fixedSeed = 1;
rng(fixedSeed,'twister');

try
    gpuDevice([]);
catch
end

%% ---------------- BEST HYPERPARAMETERS FROM ANN GRID SEARCH ----------------
% แก้ 3 ค่านี้ให้ตรงกับ BEST_RESULT จาก ANN Grid Search
numHL     = 5;
numNeuron = 64;
learnRate = 0.0005;

maxIter = 200000;              % จำนวน iteration สูงสุด (เผื่อไว้)
plotEvery = 100;

%% ---------------- EARLY STOPPING SETTING (PINN STYLE) ----------------
% หยุดเมื่อผลต่างของ Total Loss ระหว่างรอบที่ห่างกัน checkEvery
% มีค่าน้อยกว่า earlyStopTol ต่อเนื่องครบ patience ครั้ง
checkEvery   = 5000;
earlyStopTol = 1e-5;
patience     = 1;
minIter      = checkEvery;
stopCounter  = 0;
actualIter   = maxIter;

lambda_data_total = 0.005;
lambda_reg = 5e-7;

%% ============================================================
% 2) OUTPUT FOLDER
% ============================================================
outFigDir = 'Figures_ANN_N_S';
if ~exist(outFigDir,'dir')
    mkdir(outFigDir);
end

videoFile = fullfile(outFigDir,'Live_Training_ANN_N_S.mp4');

if saveVideo
    v = VideoWriter(videoFile,'MPEG-4');
    v.FrameRate = 5;
    v.Quality   = 100;
    open(v);
end

%% ============================================================
% 3) LOAD DATA
% ============================================================
Tr = readtable(trainFile,'VariableNamingRule','preserve');
Te = readtable(testFile,'VariableNamingRule','preserve');

Tr.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Tr.Properties.VariableNames));
Te.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Te.Properties.VariableNames));

X_tr = [Tr.D Tr.t Tr.fy Tr.fc Tr.L_D];
y_tr = Tr.N;

X_te = [Te.D Te.t Te.fy Te.fc Te.L_D];
y_te = Te.N;

%% ============================================================
% 4) NORMALIZATION
% ============================================================
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
Nphys  = Ntrain;   % ใช้ train ทั้งหมด บันทึกไว้เฉย ๆ

%% ============================================================
% 5) NEURAL NETWORK
% ============================================================
rng(fixedSeed,'twister');

layers = [
    featureInputLayer(5,'Normalization','none','Name','input')
];

for L = 1:numHL
    layers = [
        layers
        fullyConnectedLayer(numNeuron,'Name',sprintf('fc%d',L))
        tanhLayer('Name',sprintf('tanh%d',L))
    ];
end

layers = [
    layers
    fullyConnectedLayer(1,'Name','output')
];

net = dlnetwork(layerGraph(layers));

fprintf('\n================ ANN SETTING ================\n');
fprintf('Hidden Layers = %d\n',numHL);
fprintf('Neurons/Layer = %d\n',numNeuron);
fprintf('Learning Rate = %.1e\n',learnRate);
fprintf('Fixed Seed    = %d\n',fixedSeed);
fprintf('Training Data = %d samples\n',Ntrain);
fprintf('=============================================\n');

%% ============================================================
% 6) TRAINING INITIALIZATION
% ============================================================
trailingAvg   = [];
trailingAvgSq = [];

lossHistory      = nan(maxIter,1);
lossDataHistory  = nan(maxIter,1);
lossRegHistory   = nan(maxIter,1);

R2_train_live_history   = nan(maxIter,1);
RMSE_train_live_history = nan(maxIter,1);

R2_test_live_history   = nan(maxIter,1);
RMSE_test_live_history = nan(maxIter,1);

EarlyStopLog = table([],[],[],[],[], ...
    'VariableNames', {'Iteration','CurrentLoss','PreviousLoss','LossDifference','StopCounter'} );

%% ============================================================
% 7) TRAINING LOOP
% ============================================================
for it = 1:maxIter

    [loss,grad,lossData,lossReg] = dlfeval(@lossFunANN, net, ...
        x_data, y_data, lambda_data_total, lambda_reg);

    [net,trailingAvg,trailingAvgSq] = adamupdate( ...
        net, grad, trailingAvg, trailingAvgSq, it, learnRate);

    lossHistory(it)     = gather(extractdata(loss));
    lossDataHistory(it) = gather(extractdata(lossData));
    lossRegHistory(it)  = gather(extractdata(lossReg));

    %% ===== LIVE TESTING PREDICTION =====
    y_pred_test_live = forward(net,x_te,'Outputs','output');
    y_pred_test_live = gather(extractdata(y_pred_test_live))';
    y_pred_test_live = y_pred_test_live .* stdy + muy;

    RMSE_test_live = sqrt(mean((y_pred_test_live - y_te).^2));
    R2_test_live   = 1 - sum((y_pred_test_live - y_te).^2) / ...
                         sum((y_te - mean(y_te)).^2);

    RMSE_test_live_history(it) = RMSE_test_live;
    R2_test_live_history(it)   = R2_test_live;

    %% ===== LIVE TRAINING PREDICTION =====
    y_pred_train_live = forward(net,x_data,'Outputs','output');
    y_pred_train_live = gather(extractdata(y_pred_train_live))';
    y_pred_train_live = y_pred_train_live .* stdy + muy;

    RMSE_train_live = sqrt(mean((y_pred_train_live - y_tr).^2));
    R2_train_live   = 1 - sum((y_pred_train_live - y_tr).^2) / ...
                          sum((y_tr - mean(y_tr)).^2);

    RMSE_train_live_history(it) = RMSE_train_live;
    R2_train_live_history(it)   = R2_train_live;

    %% ===== LIVE PLOT =====
    if showLivePlot && (mod(it,plotEvery)==0 || it==1)

        figLive = figure(1); clf;
        set(figLive,'Position',[50 100 1400 560]);

        subplot(1,2,1)
        semilogy(1:it,lossHistory(1:it),'k','LineWidth',1.6); hold on;
        semilogy(1:it,lossDataHistory(1:it),'b','LineWidth',1.3);
        semilogy(1:it,lossRegHistory(1:it),'m','LineWidth',1.3);
        hold off;
        grid on;
        xlabel('Iteration');
        ylabel('Loss');
        legend('Total Loss','Data Loss','Regularization Loss','Location','best');
        title('Training Loss (ANN)');
        set(gca,'FontSize',12);

        subplot(1,2,2)
        scatter(y_tr,y_pred_train_live,22,'filled'); hold on;
        scatter(y_te,y_pred_test_live,28,'filled');

        mn = min([y_tr; y_te; y_pred_train_live; y_pred_test_live]);
        mx = max([y_tr; y_te; y_pred_train_live; y_pred_test_live]);

        plot([mn mx],[mn mx],'k--','LineWidth',1.2);
        hold off;
        grid on;
        xlabel('Experimental N_u (kN)');
        ylabel('Predicted N_u (kN)');
        legend('Training','Testing','1:1 Line','Location','best');
        title(sprintf('Iteration %d | Train R^2=%.4f | Test R^2=%.4f', ...
            it, R2_train_live, R2_test_live));
        set(gca,'FontSize',12);

        drawnow;

        if saveVideo
            frame = getframe(figLive);
            img = frame.cdata;

            [hh,ww,~] = size(img);
            if mod(hh,2) ~= 0
                img = img(1:end-1,:,:);
            end
            if mod(ww,2) ~= 0
                img = img(:,1:end-1,:);
            end

            writeVideo(v,img);
        end
    end

    if mod(it,plotEvery)==0
        fprintf('Iter %5d | Loss %.3e | Data %.3e | Reg %.3e | Train R2 %.4f | Test R2 %.4f\n', ...
            it, lossHistory(it), lossDataHistory(it), lossRegHistory(it), ...
            R2_train_live, R2_test_live);
    end

    %% ===== EARLY STOPPING CHECK (PINN STYLE) =====
    % ตรวจทุก checkEvery iteration โดยเปรียบเทียบ Total Loss ปัจจุบัน
    % กับ Total Loss เมื่อ checkEvery iteration ก่อนหน้า
    if it >= minIter && mod(it,checkEvery)==0

        lossDiff = abs(lossHistory(it) - lossHistory(it-checkEvery+1));

        if lossDiff < earlyStopTol
            stopCounter = stopCounter + 1;
        else
            stopCounter = 0;
        end

        EarlyStopLog = [EarlyStopLog; table( ...
            it, lossHistory(it), lossHistory(it-checkEvery+1), lossDiff, stopCounter, ...
            'VariableNames', {'Iteration','CurrentLoss','PreviousLoss','LossDifference','StopCounter'} )];

        fprintf('EarlyStop Check @ Iter %d | |Loss(it)-Loss(it-%d)| = %.3e | Counter = %d/%d\n', ...
            it, checkEvery, lossDiff, stopCounter, patience);

        if stopCounter >= patience
            actualIter = it;
            fprintf('\nEARLY STOPPING activated at iteration %d because loss difference %.3e < %.3e.\n', ...
                actualIter, lossDiff, earlyStopTol);
            break;
        end
    end

    actualIter = it;
end

%% ===== TRIM HISTORY TO ACTUAL ITERATION =====
lossHistory             = lossHistory(1:actualIter);
lossDataHistory         = lossDataHistory(1:actualIter);
lossRegHistory          = lossRegHistory(1:actualIter);
R2_train_live_history   = R2_train_live_history(1:actualIter);
RMSE_train_live_history = RMSE_train_live_history(1:actualIter);
R2_test_live_history    = R2_test_live_history(1:actualIter);
RMSE_test_live_history  = RMSE_test_live_history(1:actualIter);

%% ============================================================
% 8) CLOSE VIDEO
% ============================================================
if saveVideo
    close(v);
    fprintf('\nLive training video saved: %s\n',videoFile);
end

%% ============================================================
% 9) FINAL TRAINING PREDICTION
% ============================================================
y_pred_tr = forward(net,x_data,'Outputs','output');
y_pred_tr = gather(extractdata(y_pred_tr))';
y_pred_tr = y_pred_tr .* stdy + muy;

MAE_tr  = mean(abs(y_pred_tr - y_tr));
MSE_tr  = mean((y_pred_tr - y_tr).^2);
RMSE_tr = sqrt(MSE_tr);
R2_tr   = 1 - sum((y_pred_tr - y_tr).^2) / sum((y_tr - mean(y_tr)).^2);

ratio_tr = y_pred_tr ./ y_tr;
MeanRatio_tr = mean(ratio_tr);
SDRatio_tr   = std(ratio_tr);
A20_tr = mean((ratio_tr >= 0.8) & (ratio_tr <= 1.2)) * 100;

%% ============================================================
% 10) FINAL TESTING PREDICTION
% ============================================================
y_pred = forward(net,x_te,'Outputs','output');
y_pred = gather(extractdata(y_pred))';
y_pred = y_pred .* stdy + muy;

MAE  = mean(abs(y_pred - y_te));
MSE  = mean((y_pred - y_te).^2);
RMSE = sqrt(MSE);
R2   = 1 - sum((y_pred - y_te).^2) / sum((y_te - mean(y_te)).^2);

ratio = y_pred ./ y_te;
MeanRatio = mean(ratio);
SDRatio   = std(ratio);
A20 = mean((ratio >= 0.8) & (ratio <= 1.2)) * 100;

fprintf('\n================ FINAL TRAINING RESULTS ================\n');
fprintf('MAE        = %.3f kN\n',MAE_tr);
fprintf('MSE        = %.3f\n',MSE_tr);
fprintf('RMSE       = %.3f kN\n',RMSE_tr);
fprintf('R2         = %.4f\n',R2_tr);
fprintf('A20-index  = %.2f %%\n',A20_tr);
fprintf('Mean Npred/Nexp = %.4f\n',MeanRatio_tr);
fprintf('SD   Npred/Nexp = %.4f\n',SDRatio_tr);

fprintf('\n================ FINAL TESTING RESULTS ================\n');
fprintf('MAE        = %.3f kN\n',MAE);
fprintf('MSE        = %.3f\n',MSE);
fprintf('RMSE       = %.3f kN\n',RMSE);
fprintf('R2         = %.4f\n',R2);
fprintf('A20-index  = %.2f %%\n',A20);
fprintf('Mean Npred/Nexp = %.4f\n',MeanRatio);
fprintf('SD   Npred/Nexp = %.4f\n',SDRatio);
fprintf('====================================================\n');

%% ============================================================
% 11) FINAL TRAINING PARITY PLOT
% ============================================================
figTrain = figure;
set(figTrain,'Position',[100 100 700 600]);

scatter(y_tr,y_pred_tr,35,'filled'); hold on;
mn = min([y_tr; y_pred_tr]);
mx = max([y_tr; y_pred_tr]);
plot([mn mx],[mn mx],'k--','LineWidth',1.3);
hold off;
grid on;
xlabel('Experimental N_u (kN)');
ylabel('Predicted N_u (kN)');
title(sprintf('Training Parity Plot | R^2 = %.4f | RMSE = %.2f kN',R2_tr,RMSE_tr));
set(gca,'FontSize',12);

exportgraphics(figTrain,fullfile(outFigDir,'Training_Parity_Plot.png'),'Resolution',1000);
exportgraphics(figTrain,fullfile(outFigDir,'Training_Parity_Plot.tif'),'Resolution',1000);

%% ============================================================
% 12) FINAL TESTING PARITY PLOT
% ============================================================
figTest = figure;
set(figTest,'Position',[850 100 700 600]);

scatter(y_te,y_pred,35,'filled'); hold on;
mn = min([y_te; y_pred]);
mx = max([y_te; y_pred]);
plot([mn mx],[mn mx],'k--','LineWidth',1.3);
hold off;
grid on;
xlabel('Experimental N_u (kN)');
ylabel('Predicted N_u (kN)');
title(sprintf('Testing Parity Plot | R^2 = %.4f | RMSE = %.2f kN',R2,RMSE));
set(gca,'FontSize',12);

exportgraphics(figTest,fullfile(outFigDir,'Testing_Parity_Plot.png'),'Resolution',1000);
exportgraphics(figTest,fullfile(outFigDir,'Testing_Parity_Plot.tif'),'Resolution',1000);

%% ============================================================
% 13) FINAL COMBINED TRAINING + TESTING PARITY PLOT
% ============================================================
figCombined = figure;
set(figCombined,'Position',[100 100 750 600]);

scatter(y_tr,y_pred_tr,30,'filled'); hold on;
scatter(y_te,y_pred,40,'filled');

mn = min([y_tr; y_te; y_pred_tr; y_pred]);
mx = max([y_tr; y_te; y_pred_tr; y_pred]);

plot([mn mx],[mn mx],'k--','LineWidth',1.3);
hold off;
grid on;
xlabel('Experimental N_u (kN)');
ylabel('Predicted N_u (kN)');
legend('Training','Testing','1:1 Line','Location','best');
title(sprintf('Combined Parity Plot | Train R^2 = %.4f | Test R^2 = %.4f',R2_tr,R2));
set(gca,'FontSize',12);

exportgraphics(figCombined,fullfile(outFigDir,'Combined_Parity_Plot.png'),'Resolution',1000);
exportgraphics(figCombined,fullfile(outFigDir,'Combined_Parity_Plot.tif'),'Resolution',1000);

%% ============================================================
% 14) FINAL TRAINING LOSS PLOT
% ============================================================
figLoss = figure;
set(figLoss,'Position',[100 100 750 600]);

semilogy(1:actualIter,lossHistory,'k','LineWidth',1.6); hold on;
semilogy(1:actualIter,lossDataHistory,'b','LineWidth',1.2);
semilogy(1:actualIter,lossRegHistory,'m','LineWidth',1.2);
hold off;
grid on;
xlabel('Iteration');
ylabel('Loss');
legend('Total Loss','Data Loss','Regularization Loss','Location','best');
title('Training Loss');
set(gca,'FontSize',12);

exportgraphics(figLoss,fullfile(outFigDir,'Training_Loss.png'),'Resolution',1000);
exportgraphics(figLoss,fullfile(outFigDir,'Training_Loss.tif'),'Resolution',1000);

%% ============================================================
% 15) EXPORT TRAINING AND TESTING PREDICTION RESULTS TO EXCEL
% ============================================================
excelFile = 'ANN_N_S_Prediction_Results.xlsx';

TrainResult = table( ...
    X_tr(:,1), X_tr(:,2), X_tr(:,3), X_tr(:,4), X_tr(:,5), ...
    y_tr, y_pred_tr, ...
    y_pred_tr - y_tr, ...
    y_pred_tr ./ y_tr, ...
    'VariableNames', {'D','t','Fy','Fc','L_D','N_exp','N_pred','Error','Npred_Nexp'} );

TestResult = table( ...
    X_te(:,1), X_te(:,2), X_te(:,3), X_te(:,4), X_te(:,5), ...
    y_te, y_pred, ...
    y_pred - y_te, ...
    y_pred ./ y_te, ...
    'VariableNames', {'D','t','Fy','Fc','L_D','N_exp','N_pred','Error','Npred_Nexp'} );

Metrics = table( ...
    ["Training"; "Testing"], ...
    [MAE_tr; MAE], ...
    [MSE_tr; MSE], ...
    [RMSE_tr; RMSE], ...
    [R2_tr; R2], ...
    [A20_tr; A20], ...
    [MeanRatio_tr; MeanRatio], ...
    [SDRatio_tr; SDRatio], ...
    'VariableNames', {'Dataset','MAE','MSE','RMSE','R2','A20_index','Mean_Npred_Nexp','SD_Npred_Nexp'} );

Hyperparameters = table( ...
    fixedSeed, numHL, numNeuron, learnRate, maxIter, actualIter, checkEvery, earlyStopTol, patience, lambda_reg, Nphys, ...
    'VariableNames', {'FixedSeed','HiddenLayers','Neurons','LearningRate','MaxIter','ActualIter','CheckEvery','EarlyStopTol','Patience','LambdaReg','Nphys'} );

LossHistoryTable = table( ...
    (1:actualIter)', ...
    lossHistory, ...
    lossDataHistory, ...
    lossRegHistory, ...
    R2_train_live_history, ...
    RMSE_train_live_history, ...
    R2_test_live_history, ...
    RMSE_test_live_history, ...
    'VariableNames', {'Iteration','TotalLoss','DataLoss','RegLoss', ...
    'Live_Train_R2','Live_Train_RMSE','Live_Test_R2','Live_Test_RMSE'} );

writetable(TrainResult,excelFile,'Sheet','TRAINING_PREDICTION');
writetable(TestResult,excelFile,'Sheet','TESTING_PREDICTION');
writetable(Metrics,excelFile,'Sheet','METRICS');
writetable(Hyperparameters,excelFile,'Sheet','HYPERPARAMETERS');
writetable(LossHistoryTable,excelFile,'Sheet','LOSS_HISTORY');
writetable(EarlyStopLog,excelFile,'Sheet','EARLY_STOP_LOG');

fprintf('\nFigures saved in folder: %s\n',outFigDir);
fprintf('Prediction results exported to: %s\n',excelFile);

%% ============================================================
% 16) SAVE MODEL
% ============================================================
save('cfst_model_N_S_ANN.mat', ...
    'net','muX','stdX','muy','stdy', ...
    'fixedSeed','numHL','numNeuron', ...
    'lambda_data_total','lambda_reg', ...
    'maxIter','actualIter','checkEvery','earlyStopTol','patience','learnRate','Nphys', ...
    'MAE_tr','MSE_tr','RMSE_tr','R2_tr','A20_tr','MeanRatio_tr','SDRatio_tr', ...
    'MAE','MSE','RMSE','R2','A20','MeanRatio','SDRatio', ...
    'lossHistory','lossDataHistory','lossRegHistory', ...
    'R2_train_live_history','RMSE_train_live_history', ...
    'R2_test_live_history','RMSE_test_live_history');

fprintf('\nModel saved: cfst_model_N_S_ANN.mat\n');
fprintf('Total time = %.2f sec\n',toc(tTotal));

%% ============================================================
% 17) ANN LOSS FUNCTION
% ============================================================
function [loss,grad,lossData,lossReg] = lossFunANN(net,x,y,lambda_data_total,lambda_reg)

    yp = forward(net,x,'Outputs','output');
    lossData = mean((yp - y).^2,'all');

    reg = dlarray(0);
    vals = net.Learnables.Value;

    for i = 1:numel(vals)
        reg = reg + sum(vals{i}.^2,'all');
    end

    lossReg = lambda_reg .* reg;

    loss = lambda_data_total .* lossData + lossReg;

    grad = dlgradient(loss,net.Learnables);

end