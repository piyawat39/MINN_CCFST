clear; clc; close all;
tTotal = tic;
%% version 8 
%% ============================================================
% 1) SETTINGS
% ============================================================
showLivePlot = true;
saveVideo    = true ;
trainFile = 'N_S_Training_DATASET_75.xlsx';
testFile  = 'N_S_Testing_DATASET_25.xlsx';

%% ---------------- REPRODUCIBLE SETTING ----------------
fixedSeed = 1;
rng(fixedSeed,'twister');

% ใช้ CPU เพื่อให้ผลนิ่งที่สุด
try
    gpuDevice([]);
catch
end

%% ---------------- BEST HYPERPARAMETERS FROM GRID SEARCH ----------------

learnRate = 0.00005;
numNeuron = 128;
numHL     = 4;

numIter = 100000;
delta   = 0.01; % 1% change 

lambda_data_total = 0.005;
lambda_phys_total = 1;

lambda_yu    = 1;   % Yu trend loss weight inside physics loss
lambda_liang = 1;   % Liang trend loss weight inside physics loss
lambda_ld    = 1;   % L/D constraint loss: only L/D increases -> Nu should not increase

lambda_reg = 1e-7;

plotEvery = 1000;

%% ---------------- EARLY STOPPING SETTING ----------------
% Check total loss change every checkEvery iterations.
% Stop when abs(TotalLoss_now - TotalLoss_previous_check) < lossTol.
checkEvery   = 5000;
lossTol      = 1e-5;
minIterCheck = 10000;

%% ============================================================
% 2) OUTPUT FOLDER
% ============================================================
outFigDir = 'Figures_MINN_N_S';
if ~exist(outFigDir,'dir')
    mkdir(outFigDir);
end

videoFile = fullfile(outFigDir,'Live_Training_MINN_N_S.mp4');

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

minXn = min(Xn_tr,[],1);
maxXn = max(Xn_tr,[],1);

%% ============================================================
% 5) PHYSICS POINTS
% ใช้ training data ทั้งหมด
% ============================================================
Ntrain = size(Xn_tr,1);
Nphys  = Ntrain;

x_phys = dlarray(Xn_tr','CB');

%% ============================================================
% 6) NEURAL NETWORK
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

fprintf('\n================ NETWORK SETTING ================\n');
fprintf('Hidden Layers = %d\n',numHL);
fprintf('Neurons/Layer = %d\n',numNeuron);
fprintf('Learning Rate = %.1e\n',learnRate);
fprintf('Nphys         = %d / %d training samples\n',Nphys,Ntrain);
fprintf('Fixed Seed    = %d\n',fixedSeed);
fprintf('=================================================\n');

%% ============================================================
% 7) TRAINING INITIALIZATION
% ============================================================
trailingAvg   = [];
trailingAvgSq = [];

lossHistory      = zeros(numIter,1);
lossDataHistory  = zeros(numIter,1);   % Raw data loss before lambda_data_total
lossPhysHistory  = zeros(numIter,1);   % Physics loss before lambda_phys_total, but after internal physics weights
lossRegHistory   = zeros(numIter,1);   % Already weighted by lambda_reg

% Loss contributions actually used in total loss
lossDataWeightedHistory = zeros(numIter,1);
lossPhysWeightedHistory = zeros(numIter,1);
lossRegWeightedHistory  = zeros(numIter,1);

lossYuHistory    = zeros(numIter,1);   % Raw Yu trend loss
lossLiangHistory = zeros(numIter,1);   % Raw Liang trend loss
lossLDHistory    = zeros(numIter,1);   % Raw L/D constraint loss

% Weighted physics-loss contribution to TOTAL LOSS
% Note: these are multiplied by lambda_phys_total and lambda_yu/lambda_liang/lambda_ld.
% Therefore they show the real contribution seen by the optimizer.
lossYuWeightedHistory    = zeros(numIter,1);
lossLiangWeightedHistory = zeros(numIter,1);
lossLDWeightedHistory    = zeros(numIter,1);


R2_train_live_history   = zeros(numIter,1);
RMSE_train_live_history = zeros(numIter,1);

R2_test_live_history   = zeros(numIter,1);
RMSE_test_live_history = zeros(numIter,1);

%% ---------------- EARLY STOPPING INITIALIZATION ----------------
stopIteration = numIter;
earlyStopFlag = false;
lastLossChange = NaN;

%% ============================================================
% 8) TRAINING LOOP
% ============================================================
for it = 1:numIter

    [loss,grad,lossData,lossPhysTotal,lossReg,lossYu,lossLiang,lossLD] = ...
        dlfeval(@lossFun, net, ...
        x_data, y_data, x_phys, ...
        minXn, maxXn, muX, stdX, muy, stdy, ...
        lambda_data_total, lambda_phys_total, ...
        lambda_yu, lambda_liang, lambda_ld, ...
        lambda_reg, delta);

    [net,trailingAvg,trailingAvgSq] = adamupdate( ...
        net, grad, trailingAvg, trailingAvgSq, it, learnRate);

    lossHistory(it)      = gather(extractdata(loss));
    lossDataHistory(it)  = gather(extractdata(lossData));
    lossPhysHistory(it)  = gather(extractdata(lossPhysTotal));
    lossRegHistory(it)   = gather(extractdata(lossReg));

    % Weighted contributions in total loss:
    % TotalLoss = DataContribution + PhysicsContribution + RegContribution
    lossDataWeightedHistory(it) = lambda_data_total * lossDataHistory(it);
    lossPhysWeightedHistory(it) = lambda_phys_total * lossPhysHistory(it);
    lossRegWeightedHistory(it)  = lossRegHistory(it);

    lossYuHistory(it)    = gather(extractdata(lossYu));
    lossLiangHistory(it) = gather(extractdata(lossLiang));
    lossLDHistory(it)    = gather(extractdata(lossLD));

    % Weighted physics-loss terms contributing to total loss
    lossYuWeightedHistory(it)    = lambda_phys_total * lambda_yu    * lossYuHistory(it);
    lossLiangWeightedHistory(it) = lambda_phys_total * lambda_liang * lossLiangHistory(it);
    lossLDWeightedHistory(it)    = lambda_phys_total * lambda_ld    * lossLDHistory(it);
   
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

    %% ===== LIVE PLOT + VIDEO =====
    if showLivePlot && (mod(it,plotEvery)==0 || it==1)

        figLive = figure(1); clf;
        set(figLive,'Position',[50 100 1800 560]);

        subplot(1,3,1)
        semilogy(1:it,lossHistory(1:it),'k','LineWidth',1.6); hold on;
        semilogy(1:it,lossDataWeightedHistory(1:it),'b','LineWidth',1.2);
        semilogy(1:it,lossPhysWeightedHistory(1:it),'r','LineWidth',1.2);
        semilogy(1:it,lossRegWeightedHistory(1:it),'m','LineWidth',1.2);
        hold off;
        grid on;
        xlabel('Iteration');
        ylabel('Loss');
        legend('Total Loss','\lambda_{d} Data','\lambda_{m} Mechanics','\lambda_{r} Reg','Location','best');
        title('Training Loss (MINN) - Weighted Contributions');
        set(gca,'FontSize',12);

        subplot(1,3,2)
        semilogy(1:it,lossYuHistory(1:it),'Color',[0.85 0.33 0.10],'LineWidth',1.2); hold on;
        semilogy(1:it,lossLiangHistory(1:it),'m','LineWidth',1.2);
        semilogy(1:it,lossLDHistory(1:it),'g','LineWidth',1.2);
        hold off;
        grid on;
        xlabel('Iteration');
        ylabel('Raw Mechanics Loss');
        legend('Yu et al.','Liang et al','L/D constrain','Location','best');
        title('Mechanics Loss');
        set(gca,'FontSize',12);

        subplot(1,3,3)
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

            [h,w,~] = size(img);
            if mod(h,2) ~= 0
                img = img(1:end-1,:,:);
            end
            if mod(w,2) ~= 0
                img = img(:,1:end-1,:);
            end

            writeVideo(v,img);
        end
    end

    if mod(it,plotEvery)==0
        fprintf(['Iter %5d | Loss %.3e | Data %.3e | Phys %.3e | Reg %.3e | ', ...
            'Raw[Yu %.3e Liang %.3e LD %.3e] | ', ...
            'Weighted[Yu %.3e Liang %.3e LD %.3e] | ', ...
            'Train R2 %.4f | Test R2 %.4f\n'], ...
            it, lossHistory(it), lossDataHistory(it), lossPhysHistory(it), ...
            lossRegHistory(it), ...
            lossYuHistory(it), lossLiangHistory(it), lossLDHistory(it), ...
            lossYuWeightedHistory(it), lossLiangWeightedHistory(it), lossLDWeightedHistory(it), ...
            R2_train_live, R2_test_live);
    end

    %% ===== EARLY STOPPING CHECK =====
    if mod(it,checkEvery)==0 && it >= minIterCheck

        lastLossChange = abs(lossHistory(it) - lossHistory(it-checkEvery));

        fprintf('Loss Change over last %d iterations = %.6e | Tolerance = %.6e\n', ...
            checkEvery, lastLossChange, lossTol);

        if lastLossChange < lossTol

            earlyStopFlag = true;
            stopIteration = it;

            fprintf('\n=========================================\n');
            fprintf('EARLY STOPPING TRIGGERED\n');
            fprintf('Iteration   = %d\n', stopIteration);
            fprintf('Loss Change = %.6e\n', lastLossChange);
            fprintf('Tolerance   = %.6e\n', lossTol);
            fprintf('=========================================\n');

            break;

        end

    end
end

%% ============================================================
% 9) TRIM HISTORY TO ACTUAL TRAINING ITERATIONS
% ============================================================
if stopIteration < numIter
    fprintf('\nTraining stopped early at iteration %d of %d.\n', stopIteration, numIter);
else
    fprintf('\nTraining reached maximum iteration: %d.\n', numIter);
end

lossHistory      = lossHistory(1:stopIteration);
lossDataHistory  = lossDataHistory(1:stopIteration);
lossPhysHistory  = lossPhysHistory(1:stopIteration);
lossRegHistory   = lossRegHistory(1:stopIteration);

lossDataWeightedHistory = lossDataWeightedHistory(1:stopIteration);
lossPhysWeightedHistory = lossPhysWeightedHistory(1:stopIteration);
lossRegWeightedHistory  = lossRegWeightedHistory(1:stopIteration);

lossYuHistory    = lossYuHistory(1:stopIteration);
lossLiangHistory = lossLiangHistory(1:stopIteration);
lossLDHistory    = lossLDHistory(1:stopIteration);

lossYuWeightedHistory    = lossYuWeightedHistory(1:stopIteration);
lossLiangWeightedHistory = lossLiangWeightedHistory(1:stopIteration);
lossLDWeightedHistory    = lossLDWeightedHistory(1:stopIteration);

R2_train_live_history   = R2_train_live_history(1:stopIteration);
RMSE_train_live_history = RMSE_train_live_history(1:stopIteration);

R2_test_live_history   = R2_test_live_history(1:stopIteration);
RMSE_test_live_history = RMSE_test_live_history(1:stopIteration);

numIterMax = numIter;
numIter = stopIteration;

%% ============================================================
% 10) CLOSE VIDEO
% ============================================================
if saveVideo
    close(v);
    fprintf('\nLive training video saved: %s\n',videoFile);
end

%% ============================================================
% 10) FINAL TRAINING PREDICTION
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
% 11) FINAL TESTING PREDICTION
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
% 12) FINAL TRAINING PARITY PLOT
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
% 13) FINAL TESTING PARITY PLOT
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
% 14) FINAL COMBINED TRAINING + TESTING PARITY PLOT
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
% 15) FINAL TRAINING LOSS PLOT
% ============================================================
figLoss = figure;
set(figLoss,'Position',[100 100 750 600]);

semilogy(1:numIter,lossHistory,'k','LineWidth',1.6); hold on;
semilogy(1:numIter,lossDataWeightedHistory,'b','LineWidth',1.2);
semilogy(1:numIter,lossPhysWeightedHistory,'r','LineWidth',1.2);
semilogy(1:numIter,lossRegWeightedHistory,'m','LineWidth',1.2);
hold off;
grid on;
xlabel('Iteration');
ylabel('Loss');
legend('Total Loss','\lambda_{d} Data','\lambda_{m} Mechanics','\lambda_{r} Reg','Location','best');
title('Training Loss Convergence');
set(gca,'FontSize',14);

exportgraphics(figLoss,fullfile(outFigDir,'Training_Loss.png'),'Resolution',1000);
exportgraphics(figLoss,fullfile(outFigDir,'Training_Loss.tif'),'Resolution',1000);

%% ============================================================
% 16) FINAL PHYSICS LOSS BREAKDOWN PLOT
% ============================================================
figPhys = figure;
set(figPhys,'Position',[100 100 750 600]);

semilogy(1:numIter,lossYuHistory,'Color',[0.85 0.33 0.10],'LineWidth',1.2); hold on;
semilogy(1:numIter,lossLiangHistory,'m','LineWidth',1.2);
semilogy(1:numIter,lossLDHistory,'g','LineWidth',1.2);
hold off;
grid on;
xlabel('Iteration');
ylabel('Mechanics Loss');
legend('Yu et al.','Liang et al.','L/D constraint','Location','best');
title('Mechanics Loss Breakdown');
set(gca,'FontSize',14);

exportgraphics(figPhys,fullfile(outFigDir,'Mechanics_Loss_Breakdown.png'),'Resolution',1000);
exportgraphics(figPhys,fullfile(outFigDir,'Mechanics_Loss_Breakdown.tif'),'Resolution',1000);

%% ============================================================
% 17) FINAL LIVE STYLE PLOT
% ============================================================
figLiveFinal = figure;
set(figLiveFinal,'Position',[50 100 1800 560]);

subplot(1,3,1)
semilogy(1:numIter,lossHistory,'k','LineWidth',1.6); hold on;
semilogy(1:numIter,lossDataWeightedHistory,'b','LineWidth',1.2);
semilogy(1:numIter,lossPhysWeightedHistory,'r','LineWidth',1.2);
semilogy(1:numIter,lossRegWeightedHistory,'m','LineWidth',1.2);
hold off;
grid on;
xlabel('Iteration');
ylabel('Loss');
legend('Total Loss','\lambda_{d} Data','\lambda_{m} Physics','\lambda_{r} Reg','Location','best');
title('Training Loss (MINN) ');
set(gca,'FontSize',12);

subplot(1,3,2)
semilogy(1:numIter,lossYuHistory,'Color',[0.85 0.33 0.10],'LineWidth',1.2); hold on;
semilogy(1:numIter,lossLiangHistory,'m','LineWidth',1.2);
semilogy(1:numIter,lossLDHistory,'g','LineWidth',1.2);
hold off;
grid on;
xlabel('Iteration');
ylabel('Mechanics Loss');
legend('Yu at el.','Liang at el.','L/D constrain','Location','best');
title('Mechanics Loss');
set(gca,'FontSize',12);

subplot(1,3,3)
scatter(y_tr,y_pred_tr,22,'filled'); hold on;
scatter(y_te,y_pred,28,'filled');
mn = min([y_tr; y_te; y_pred_tr; y_pred]);
mx = max([y_tr; y_te; y_pred_tr; y_pred]);
plot([mn mx],[mn mx],'k--','LineWidth',1.2);
hold off;
grid on;
xlabel('Experimental N_u (kN)');
ylabel('Predicted N_u (kN)');
legend('Training','Testing','1:1 Line','Location','best');
title(sprintf('Final | Train R^2=%.4f | Test R^2=%.4f',R2_tr,R2));
set(gca,'FontSize',12);

exportgraphics(figLiveFinal,fullfile(outFigDir,'Final_Live_Training_Plot.png'),'Resolution',1000);
exportgraphics(figLiveFinal,fullfile(outFigDir,'Final_Live_Training_Plot.tif'),'Resolution',1000);

%% ============================================================
% 18) EXPORT TRAINING AND TESTING PREDICTION RESULTS TO EXCEL
% ============================================================
excelFile = 'MINN_N_S_Prediction_Results_ver2.xlsx';

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

LossWeights = table( ...
    ["lambda_data_total"; "lambda_phys_total"; ...
     "lambda_yu"; "lambda_liang"; "lambda_ld"; "lambda_reg"], ...
    [lambda_data_total; lambda_phys_total; ...
     lambda_yu; lambda_liang; lambda_ld; lambda_reg], ...
    'VariableNames', {'Weight_Name','Value'} );

Hyperparameters = table( ...
    fixedSeed, ...
    numHL, ...
    numNeuron, ...
    learnRate, ...
    numIterMax, ...
    numIter, ...
    delta, ...
    Nphys, ...
    checkEvery, ...
    lossTol, ...
    minIterCheck, ...
    earlyStopFlag, ...
    lastLossChange, ...
    'VariableNames', {'FixedSeed','HiddenLayers','Neurons','LearningRate', ...
    'MaxNumIter','ActualNumIter','Delta','Nphys','CheckEvery', ...
    'LossTolerance','MinIterCheck','EarlyStopFlag','LastLossChange'} );

%% ============================================================
% LOSS HISTORY TABLE
%% ============================================================
LossHistoryTable = table( ...
    (1:numIter)', ...
    lossHistory, ...
    lossDataHistory, ...
    lossPhysHistory, ...
    lossRegHistory, ...
    lossDataWeightedHistory, ...
    lossPhysWeightedHistory, ...
    lossRegWeightedHistory, ...
    lossYuHistory, ...
    lossLiangHistory, ...
    lossLDHistory, ...
    lossYuWeightedHistory, ...
    lossLiangWeightedHistory, ...
    lossLDWeightedHistory, ...
    R2_train_live_history, ...
    RMSE_train_live_history, ...
    R2_test_live_history, ...
    RMSE_test_live_history, ...
    'VariableNames',{ ...
    'Iteration', ...
    'TotalLoss', ...
    'DataLoss', ...
    'PhysicsLoss', ...
    'RegLoss', ...
    'DataLoss_WeightedContribution', ...
    'PhysicsLoss_WeightedContribution', ...
    'RegLoss_WeightedContribution', ...
    'YuDeltaTrendLoss_Raw', ...
    'LiangDeltaTrendLoss_Raw', ...
    'LDConstraintLoss_Raw', ...
    'YuDeltaTrendLoss_Weighted', ...
    'LiangDeltaTrendLoss_Weighted', ...
    'LDConstraintLoss_Weighted', ...
    'Live_Train_R2', ...
    'Live_Train_RMSE', ...
    'Live_Test_R2', ...
    'Live_Test_RMSE'} );

writetable(TrainResult,excelFile,'Sheet','TRAINING_PREDICTION');
writetable(TestResult,excelFile,'Sheet','TESTING_PREDICTION');
writetable(Metrics,excelFile,'Sheet','METRICS');
writetable(LossWeights,excelFile,'Sheet','LOSS_WEIGHTS');
writetable(Hyperparameters,excelFile,'Sheet','HYPERPARAMETERS');
writetable(LossHistoryTable,excelFile,'Sheet','LOSS_HISTORY');

fprintf('\nFigures saved in folder: %s\n',outFigDir);
fprintf('Prediction results exported to: %s\n',excelFile);

%% ============================================================
% 19) SAVE MODEL
% ============================================================
save('cfst_model_N_S_MINN_ver2.mat', ...
    'net','muX','stdX','muy','stdy', ...
    'fixedSeed','numHL','numNeuron', ...
    'lambda_data_total','lambda_phys_total', ...
    'lambda_yu','lambda_liang','lambda_ld','lambda_reg', ...
    'delta','numIter','numIterMax','learnRate','Nphys', ...
    'checkEvery','lossTol','minIterCheck','earlyStopFlag','lastLossChange', ...
    'MAE_tr','MSE_tr','RMSE_tr','R2_tr','A20_tr','MeanRatio_tr','SDRatio_tr', ...
    'MAE','MSE','RMSE','R2','A20','MeanRatio','SDRatio', ...
    'lossHistory','lossDataHistory','lossPhysHistory','lossRegHistory', ...
    'lossDataWeightedHistory','lossPhysWeightedHistory','lossRegWeightedHistory', ...
    'lossYuHistory','lossLiangHistory','lossLDHistory', ...
    'lossYuWeightedHistory','lossLiangWeightedHistory','lossLDWeightedHistory', ...
    'R2_train_live_history','RMSE_train_live_history', ...
    'R2_test_live_history','RMSE_test_live_history');

fprintf('\nModel saved: cfst_model_N_S_ver2.mat\n');
fprintf('Total time = %.2f sec\n',toc(tTotal));


function [loss,grad,lossData,lossPhysTotal,lossReg,lossYu,lossLiang,lossLD] = ...
    lossFun(net,x,y,xp, ...
    minX,maxX,muX,stdX,muy,stdy, ...
    lambda_data_total, lambda_phys_total, ...
    lambda_yu, lambda_liang, lambda_ld,  ...
    lambda_reg, d)

    %#ok<INUSD>  % d is kept for compatibility with the main script

    %% ============================================================
    % A) DATA LOSS
    %% ============================================================
    yp_data = forward(net,x,'Outputs','output');
    lossData = mean((yp_data - y).^2,'all');

    %% ============================================================
    % B) REAL-SCALE PHYSICS INPUT
    %% ============================================================
    X_real = extractdata(xp)';
    X_real = X_real .* stdX + muX;

    Xmin_real = minX .* stdX + muX;
    Xmax_real = maxX .* stdX + muX;

    %% ============================================================
    % BASE PREDICTION FOR PHYSICS TREND LOSSES
    %% ============================================================
    y_phys = forward(net, xp, 'Outputs','output');
    N_pred_real = y_phys .* stdy + muy;   % kN

    %% ============================================================
    % C1) YU TREND LOSS
    %
    % Concept:
    %   Yu equation is used only as a trend reference.
    %   The network is NOT forced to match the Yu value directly.
    %
    %   If N_Yu_plus > N_Yu_base, prediction should increase.
    %   If N_Yu_minus < N_Yu_base, prediction should decrease.
    %% ============================================================
    lossYu = calc_equation_trend_loss( ...
        net, X_real, N_pred_real, Xmin_real, Xmax_real, ...
        muX, stdX, muy, stdy, 'yu', d);

    %% ============================================================
    % C2) LIANG TREND LOSS
    %
    % Concept:
    %   Liang & Fragomeni equation is used only as a trend reference.
    %   The network is NOT forced to match the Liang value directly.
    %
    %   Validity of Liang equation is checked at base/perturbed points.
    %% ============================================================
    lossLiang = calc_equation_trend_loss( ...
        net, X_real, N_pred_real, Xmin_real, Xmax_real, ...
        muX, stdX, muy, stdy, 'liang', d);


    %% ============================================================
    % C3) L/D CONSTRAINT LOSS
    %
    % Structural constraint for short/stub column dataset:
    %   When L/D increases, predicted Nu should not increase.
    %
    % This is a one-sided trend constraint only. It does NOT force the
    % model to match any equation value.
    %% ============================================================
    lossLD = calc_LD_constraint_loss( ...
        net, X_real, N_pred_real, Xmin_real, Xmax_real, ...
        muX, stdX, muy, stdy, d);

    %% ============================================================
    % D) COMBINE PHYSICS LOSSES
    %% ============================================================
    lossPhysTotal = ...
        lambda_yu    .* lossYu    + ...
        lambda_liang .* lossLiang + ...
        lambda_ld    .* lossLD;
      
    %% ============================================================
    % E) L2 REGULARIZATION
    %% ============================================================
    reg = dlarray(0);
    vals = net.Learnables.Value;

    for i = 1:numel(vals)
        reg = reg + sum(vals{i}.^2,'all');
    end

    lossReg = lambda_reg .* reg;

    %% ============================================================
    % F) TOTAL LOSS
    %% ============================================================
    loss = ...
        lambda_data_total .* lossData + ...
        lambda_phys_total .* lossPhysTotal + ...
        lossReg;

    grad = dlgradient(loss,net.Learnables);

end


%% ============================================================
% HELPER FUNCTION: L/D constraint loss
% หลักการ:
%   L/D เพิ่ม  -> Nu ไม่ควรเพิ่ม
% ใช้เฉพาะทิศทางเดียว ไม่ได้กำหนดค่าความชัน
% ไม่บังคับกรณี L/D ลด
%% ============================================================
function lossLD = calc_LD_constraint_loss( ...
    net, X_real, y_base_real, Xmin_real, Xmax_real, ...
    muX, stdX, muy, stdy, d)

    pos = @(z) max(0,z);

    kLD = 5;                 % column 5 = L/D
    dLD = d;                 % absolute perturbation, not percentage

    X_plus = X_real;

    % perturb L/D only: increase side only

    X_plus(:,kLD) = X_plus(:,kLD) .* (1 + dLD);

    % clamp within training range
    X_plus(:,kLD) = min(max(X_plus(:,kLD), Xmin_real(kLD)), Xmax_real(kLD));

    % normalize back to network input
    x_plus = dlarray(((X_plus - muX)./stdX)','CB');

    % prediction in real scale
    y_plus_real = forward(net, x_plus, 'Outputs','output') .* stdy + muy;

    % L/D increase should not make Nu increase
    % penalty occurs only when y_plus_real > y_base_real
    loss_plus = mean(pos(y_plus_real - y_base_real).^2,'all');

    % normalize by output scale to keep magnitude stable
    yScale = max(mean(abs(extractdata(y_base_real))), eps);
    lossLD = loss_plus ./ (yScale^2);

end

%% ============================================================
% HELPER FUNCTION: Equation-based ABSOLUTE-DELTA trend loss
% ใช้สมการ Yu หรือ Liang เป็นตัวบอก "แนวโน้มเชิงสัดส่วน" เท่านั้น
% ไม่ได้บังคับให้ค่าทำนายเท่ากับค่าสมการโดยตรง
%
% หลักการ:
%   dEq_rel   = (N_eq_perturbed - N_eq_base) / N_eq_base
%   dPred_rel = (N_pred_perturbed - N_pred_base) / N_pred_base
%
%   ถ้า Yu/Liang บอกว่าเพิ่ม 5% โมเดลก็ควรเพิ่มใกล้เคียง 5%
%   ถ้าเพิ่มน้อย/มากเกินไป หรือทิศทางผิด จะเกิด penalty
%% ============================================================
function lossTrend = calc_equation_trend_loss( ...
    net, X_real, y_base_real, Xmin_real, Xmax_real, ...
    muX, stdX, muy, stdy, equationName, d)

    trendVars = 1:4;      % D, t, fy, fc
    dTrend    = d;        
    lossTrend = dlarray(0);
    nTerm = 0;

    [N_base, valid_base] = calc_equation_value(X_real, equationName);

    for k = trendVars

        X_plus  = X_real;
        X_minus = X_real;

        % perturb one variable in real scale using absolute delta
        % Example: if d = 0.10, D becomes D + 0.10, fy becomes fy + 0.10, etc.
        X_plus(:,k)  = X_plus(:,k)  .* (1 + dTrend);
        X_minus(:,k) = X_minus(:,k) .* (1 - dTrend);

        % clamp within training range
        X_plus(:,k)  = min(max(X_plus(:,k),  Xmin_real(k)), Xmax_real(k));
        X_minus(:,k) = min(max(X_minus(:,k), Xmin_real(k)), Xmax_real(k));

        % normalize back to network input
        x_plus  = dlarray(((X_plus  - muX)./stdX)','CB');
        x_minus = dlarray(((X_minus - muX)./stdX)','CB');

        % network prediction in real scale
        y_plus_real  = forward(net, x_plus,  'Outputs','output') .* stdy + muy;
        y_minus_real = forward(net, x_minus, 'Outputs','output') .* stdy + muy;

        % equation values at perturbed points
        [N_plus,  valid_plus]  = calc_equation_value(X_plus,  equationName);
        [N_minus, valid_minus] = calc_equation_value(X_minus, equationName);

        %% ----- plus direction: absolute-difference trend -----
        validP = valid_base & valid_plus & ...
                 isfinite(N_base) & isfinite(N_plus) & ...
                 N_base > 0 & N_plus > 0;

        if any(validP)

            N0 = dlarray(N_base(validP)','CB');
            N1 = dlarray(N_plus(validP)','CB');

            Y0 = y_base_real(:,validP');
            Y1 = y_plus_real(:,validP');

            % Absolute change from equation and PINN prediction
            dEq = N1 - N0;
            dPr = Y1 - Y0;

            % Normalize by one common output scale to avoid kN-scale domination
            scaleN = max(mean(abs(N_base(validP))), eps);

            lossTrend = lossTrend + mean(((dPr - dEq)./scaleN).^2,'all');
            nTerm = nTerm + 1;
        end

        %% ----- minus direction: absolute-difference trend -----
        validM = valid_base & valid_minus & ...
                 isfinite(N_base) & isfinite(N_minus) & ...
                 N_base > 0 & N_minus > 0;

        if any(validM)

            N0 = dlarray(N_base(validM)','CB');
            N1 = dlarray(N_minus(validM)','CB');

            Y0 = y_base_real(:,validM');
            Y1 = y_minus_real(:,validM');

            % Absolute change from equation and PINN prediction
            dEq = N1 - N0;
            dPr = Y1 - Y0;

            % Normalize by one common output scale to avoid kN-scale domination
            scaleN = max(mean(abs(N_base(validM))), eps);

            lossTrend = lossTrend + mean(((dPr - dEq)./scaleN).^2,'all');
            nTerm = nTerm + 1;
        end

    end

    if nTerm > 0
        lossTrend = lossTrend ./ nTerm;
    else
        lossTrend = dlarray(0);
    end

end

%% ============================================================
% HELPER FUNCTION: Select equation value
%% ============================================================
function [N_eq, valid] = calc_equation_value(X_real, equationName)

    switch lower(equationName)
        case 'yu'
            N_eq = calc_N_yu(X_real);
            valid = isfinite(N_eq) & N_eq > 0;

        case 'liang'
            [N_eq, valid] = calc_N_liang(X_real);
            valid = valid & isfinite(N_eq) & N_eq > 0;

        otherwise
            error('Unknown equationName: %s', equationName);
    end

end

%% ============================================================
% HELPER FUNCTION: Squash load Nsq = Ac*fc + As*fy
% Output unit: kN
%% ============================================================
function Nsq = calc_Nsq(X_real)

    D  = max(X_real(:,1), eps);
    t  = max(X_real(:,2), eps);
    fy = max(X_real(:,3), eps);
    fc = max(X_real(:,4), eps);

    Di = max(D - 2.*t, eps);

    Ac = pi/4 .* Di.^2;
    As = pi/4 .* (D.^2 - Di.^2);

    Nsq = (Ac .* fc + As .* fy) ./ 1000;

    Nsq(~isfinite(Nsq)) = 0;

end

%% ============================================================
% HELPER FUNCTION: Yu unified formula
% Output unit: kN
%% ============================================================
function N_Yu = calc_N_yu(X_real)

    D  = max(X_real(:,1), eps);
    t  = max(X_real(:,2), eps);
    fy = max(X_real(:,3), eps);
    fc = max(X_real(:,4), eps);

    Di = max(D - 2.*t, eps);

    As = pi/4 .* (D.^2 - Di.^2);
    Ac = pi/4 .* Di.^2;

    valid = fc > 0 & fy > 0 & Ac > 0 & As > 0;

    N_Yu = nan(size(D));

    if any(valid)

        fck = 0.8 .* fc(valid);

        Asc = As(valid) + Ac(valid);
        beta = As(valid) ./ Asc;

        Omega = ones(size(Ac(valid)));

        xi_sc = (As(valid).*fy(valid)) ./ (Ac(valid).*fck);

        etaYu = (Omega.*xi_sc) ./ ...
            ((2.0.*Omega + 0.05.*xi_sc + ...
            (0.2.*fck./fy(valid) - 0.05).*xi_sc.*Omega) ...
            .* (Omega + xi_sc));

        fsc = (1 + etaYu) .* ...
            ((1 - beta).*fck + beta.*fy(valid));

        N_Yu(valid) = fsc .* Asc ./ 1000;

    end

end

%% ============================================================
% HELPER FUNCTION: Liang & Fragomeni formula
% Output unit: kN
%% ============================================================
function [N_Liang, validLiang] = calc_N_liang(X_real)

    D  = max(X_real(:,1), eps);
    t  = max(X_real(:,2), eps);
    fy = max(X_real(:,3), eps);
    fc = max(X_real(:,4), eps);

    Di = max(D - 2.*t, eps);

    As = pi/4 .* (D.^2 - Di.^2);
    Ac = pi/4 .* Di.^2;

    Dt = D ./ t;
    ratio_fc_fy = fc ./ fy;

    validLiang = ...
        Dt > 0 & ...
        Dt <= 150 & ...
        fc > 0 & ...
        fy > 0 & ...
        t > 0 & ...
        Di > 0 & ...
        ratio_fc_fy >= 0.04 & ...
        ratio_fc_fy <= 0.20;

    N_Liang = nan(size(D));

    if any(validLiang)

        Dv  = D(validLiang);
        tv  = t(validLiang);
        fyv = fy(validLiang);
        fcv = fc(validLiang);
        Div = Di(validLiang);

        Asv = As(validLiang);
        Acv = Ac(validLiang);

        Dtv = Dv ./ tv;

        gamma_c = 1.85 .* Div.^(-0.135);
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
                (2.*t1 ./ max(D1 - 2.*t1,eps)) .* fy1;

            frp(idx1) = max(frp(idx1),0);

        end

        if any(idx2)

            frp(idx2) = ...
                (0.006241 - 0.0000357 .* Dtv(idx2)) .* fyv(idx2);

            frp(idx2) = max(frp(idx2),0);

        end

        N_Liang(validLiang) = ...
            ((gamma_c.*fcv + 4.1.*frp).*Acv + ...
            gamma_s.*fyv.*Asv) ./ 1000;

    end

end