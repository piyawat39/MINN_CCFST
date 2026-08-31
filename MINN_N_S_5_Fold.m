clear; clc; close all;
tTotal = tic;

%% ============================================================
% MINN: 5-FOLD CROSS-VALIDATION
% - Performs 5-fold CV on the original training set only.
% - Keeps the original testing set completely outside all folds.
% - Each training sample is used for validation exactly once.
% - Normalization is fitted using the training folds only.
% - Final testing prediction is the mean prediction from the 5 fold models.
%% ============================================================

%% 1) SETTINGS
trainFile = 'N_S_Training_DATASET_75.xlsx';
testFile  = 'N_S_Testing_DATASET_25.xlsx';
K = 5;
fixedSeed = 1;

learnRate = 1.982e-05;
numNeuron = 93;
numHL = 4;
numIterMax = 100000;
delta = 0.01;

lambda_data_total = 0.001014;
lambda_phys_total = 1;
lambda_yu = 1;
lambda_liang = 1;
lambda_ld = 1;
lambda_reg = 1e-7;

checkEvery = 5000;
lossTol = 1e-5;
minIterCheck = 10000;
printEvery = 1000;

% ---------------- LIVE PLOT OPTIONS ----------------
showLivePlot = true;       % true = display live plot; false = hide it
saveVideo = false;         % true = record MP4; false = do not record
livePlotEvery = 1000;      % plot/video update interval (iterations)

outDir = 'Results_MINN_N_S_5Fold';
if ~exist(outDir,'dir'), mkdir(outDir); end

rng(fixedSeed,'twister');
try, gpuDevice([]); catch, end

%% 2) LOAD TRAINING AND INDEPENDENT TESTING DATA
Tr0 = readtable(trainFile,'VariableNamingRule','preserve');
Te0 = readtable(testFile,'VariableNamingRule','preserve');
Tr0.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Tr0.Properties.VariableNames));
Te0.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Te0.Properties.VariableNames));
X = [Tr0.D Tr0.t Tr0.fy Tr0.fc Tr0.L_D];
y = Tr0.N;
Xtest = [Te0.D Te0.t Te0.fy Te0.fc Te0.L_D];
ytest = Te0.N;
n = size(X,1);
nTest = size(Xtest,1);

assert(n >= K,'The number of observations must be at least K.');
assert(all(isfinite(X),'all') && all(isfinite(y)), ...
    'The dataset contains NaN or Inf values.');

% Requires Statistics and Machine Learning Toolbox.
cv = cvpartition(n,'KFold',K);

metricNames = {'R2','MAE','MSE','RMSE','MAPE_percent','A20_percent', ...
    'Mean_Npred_Nexp','SD_Npred_Nexp'};
trainMetrics = nan(K,numel(metricNames));
valMetrics = nan(K,numel(metricNames));
actualIterations = nan(K,1);
earlyStopped = false(K,1);
trainTime = nan(K,1);       % Training loop only (seconds)
foldTotalTime = nan(K,1);   % Setup + training + prediction (seconds)
livePlotTime = zeros(K,1);  % Plotting overhead excluded from trainTime

oofPred = nan(n,1);
foldID = nan(n,1);
testPredByFold = nan(nTest,K);

fprintf('\n=====================================================\n');
fprintf('MINN %d-FOLD CROSS-VALIDATION | Training N = %d\n',K,n);
fprintf('Each round trains on %d folds and validates on 1 fold.\n',K-1);
fprintf('Independent testing N = %d (excluded from every fold).\n',nTest);
fprintf('=====================================================\n');

%% 3) CROSS-VALIDATION LOOP
for fold = 1:K
    tFold = tic;
    foldDir = fullfile(outDir,sprintf('Fold_%02d',fold));
    if ~exist(foldDir,'dir'), mkdir(foldDir); end
    idxTr = training(cv,fold);
    idxVa = test(cv,fold);

    Xtr = X(idxTr,:); ytr = y(idxTr);
    Xva = X(idxVa,:); yva = y(idxVa);

    % Fold-specific normalization prevents validation leakage.
    muX = mean(Xtr,1);
    stdX = std(Xtr,0,1); stdX(stdX==0) = 1;
    muy = mean(ytr);
    stdy = std(ytr); if stdy==0, stdy=1; end

    XnTr = (Xtr-muX)./stdX;
    XnVa = (Xva-muX)./stdX;
    XnTest = (Xtest-muX)./stdX;
    ynTr = (ytr-muy)./stdy;

    xData = dlarray(XnTr','CB');
    yData = dlarray(ynTr','CB');
    xPhys = xData;
    xVal = dlarray(XnVa','CB');
    xTest = dlarray(XnTest','CB');
    minXn = min(XnTr,[],1);
    maxXn = max(XnTr,[],1);

    % A different but reproducible initialization for each fold.
    rng(fixedSeed + fold - 1,'twister');
    layers = featureInputLayer(5,'Normalization','none','Name','input');
    for j = 1:numHL
        layers = [layers
            fullyConnectedLayer(numNeuron,'Name',sprintf('fc%d',j))
            tanhLayer('Name',sprintf('tanh%d',j))]; %#ok<AGROW>
    end
    layers = [layers; fullyConnectedLayer(1,'Name','output')];
    net = dlnetwork(layerGraph(layers));

    trailingAvg = []; trailingAvgSq = [];
    lossHistory = nan(numIterMax,1);
    lossDataHistory = nan(numIterMax,1);
    lossPhysHistory = nan(numIterMax,1);
    lossRegHistory = nan(numIterMax,1);
    lossDataWeightedHistory = nan(numIterMax,1);
    lossPhysWeightedHistory = nan(numIterMax,1);
    lossRegWeightedHistory = nan(numIterMax,1);
    lossYuHistory = nan(numIterMax,1);
    lossLiangHistory = nan(numIterMax,1);
    lossLDHistory = nan(numIterMax,1);
    stopIteration = numIterMax;

    fprintf('\n---------------- FOLD %d/%d ----------------\n',fold,K);
    fprintf('Training = %d | Validation = %d | Seed = %d\n', ...
        sum(idxTr),sum(idxVa),fixedSeed+fold-1);

    % Create a graphics window when either display or recording is enabled.
    % When only video is enabled, the window remains invisible.
    if showLivePlot || saveVideo
        if showLivePlot, figVisibility='on'; else, figVisibility='off'; end
        figLive = figure('Name',sprintf('MINN Live Training - Fold %d',fold), ...
            'NumberTitle','off','Visible',figVisibility, ...
            'Position',[50 100 1800 560]);
    else
        figLive = [];
    end

    if saveVideo
        videoFile = fullfile(foldDir,sprintf('Live_Training_Fold_%02d.mp4',fold));
        videoWriter = VideoWriter(videoFile,'MPEG-4');
        videoWriter.FrameRate = 5;
        videoWriter.Quality = 100;
        open(videoWriter);
    else
        videoWriter = [];
    end

    % Start timer immediately before optimization. This excludes data
    % preparation, prediction, plotting, and file export.
    tTrain = tic;
    for it = 1:numIterMax
        [loss,grad,lossData,lossPhys,lossReg,lossYu,lossLiang,lossLD] = ...
            dlfeval(@lossFun,net,xData,yData,xPhys, ...
            minXn,maxXn,muX,stdX,muy,stdy, ...
            lambda_data_total,lambda_phys_total,lambda_yu, ...
            lambda_liang,lambda_ld,lambda_reg,delta);

        [net,trailingAvg,trailingAvgSq] = adamupdate( ...
            net,grad,trailingAvg,trailingAvgSq,it,learnRate);
        lossHistory(it) = gather(extractdata(loss));
        lossDataHistory(it) = gather(extractdata(lossData));
        lossPhysHistory(it) = gather(extractdata(lossPhys));
        lossRegHistory(it) = gather(extractdata(lossReg));
        lossDataWeightedHistory(it) = lambda_data_total*lossDataHistory(it);
        lossPhysWeightedHistory(it) = lambda_phys_total*lossPhysHistory(it);
        lossRegWeightedHistory(it) = lossRegHistory(it);
        lossYuHistory(it) = gather(extractdata(lossYu));
        lossLiangHistory(it) = gather(extractdata(lossLiang));
        lossLDHistory(it) = gather(extractdata(lossLD));

        %% ===== LIVE PLOT FOR THE CURRENT FOLD =====
        if (showLivePlot || saveVideo) && isgraphics(figLive) && ...
                (it==1 || mod(it,livePlotEvery)==0 || it==numIterMax)
            tPlot = tic;

            predTrLive = predictReal(net,xData,muy,stdy);
            predVaLive = predictReal(net,xVal,muy,stdy);
            mTrLive = calcMetrics(ytr,predTrLive);
            mVaLive = calcMetrics(yva,predVaLive);

            figure(figLive); clf(figLive);
            tiledlayout(figLive,1,3,'TileSpacing','compact','Padding','compact');

            nexttile;
            semilogy(1:it,lossHistory(1:it),'k','LineWidth',1.6); hold on;
            semilogy(1:it,lossDataWeightedHistory(1:it),'b','LineWidth',1.2);
            semilogy(1:it,lossPhysWeightedHistory(1:it),'r','LineWidth',1.2);
            semilogy(1:it,lossRegWeightedHistory(1:it),'m','LineWidth',1.2);
            hold off; grid on; xlabel('Iteration'); ylabel('Loss');
            legend('Total Loss','\lambda_d Data','\lambda_m Mechanics', ...
                '\lambda_r Reg','Location','best');
            title(sprintf('Model %d/%d Training Loss',fold,K));

            nexttile;
            semilogy(1:it,lossYuHistory(1:it),'Color',[0.85 0.33 0.10], ...
                'LineWidth',1.3); hold on;
            semilogy(1:it,lossLiangHistory(1:it),'m','LineWidth',1.3);
            semilogy(1:it,lossLDHistory(1:it),'Color',[0.10 0.60 0.20], ...
                'LineWidth',1.3);
            hold off; grid on;
            xlabel('Iteration'); ylabel('mechanics loss');
            legend('Yu et al.','Liang & Fragomeni','L/D constraint', ...
                'Location','best');
            title(sprintf( ...
    'Model %d/%d Mechanics-Loss Breakdown',fold,K));

            nexttile;
            scatter(ytr,predTrLive,20,'filled'); hold on;
            scatter(yva,predVaLive,32,'filled');
            mnLive = min([ytr;yva;predTrLive;predVaLive]);
            mxLive = max([ytr;yva;predTrLive;predVaLive]);
            plot([mnLive mxLive],[mnLive mxLive],'k--','LineWidth',1.2);
            hold off; grid on;
            xlabel('Experimental N_u (kN)');
            ylabel('Predicted N_u (kN)');
            legend('Training','Validation','1:1 Line','Location','best');
            title(sprintf( ...
    'Train R^2 = %.4f | Validation R^2 = %.4f', ...
    mTrLive(1),mVaLive(1)));

            drawnow;

            if saveVideo && ~isempty(videoWriter)
                frame = getframe(figLive);
                img = frame.cdata;
                [h,w,~] = size(img);
                if mod(h,2)~=0, img=img(1:end-1,:,:); end
                if mod(w,2)~=0, img=img(:,1:end-1,:); end
                writeVideo(videoWriter,img);
            end
            livePlotTime(fold) = livePlotTime(fold) + toc(tPlot);
        end

        if mod(it,printEvery)==0
            fprintf(['Fold %d | Iter %6d | Loss %.3e | Data %.3e | ' ...
                'Phys %.3e | Reg %.3e | Raw[Yu %.3e Liang %.3e LD %.3e]\n'], ...
                fold,it,lossHistory(it),lossDataHistory(it),lossPhysHistory(it), ...
                lossRegHistory(it),lossYuHistory(it),lossLiangHistory(it), ...
                lossLDHistory(it));
        end

        if mod(it,checkEvery)==0 && it>=minIterCheck
            lossChange = abs(lossHistory(it)-lossHistory(it-checkEvery));
            if lossChange < lossTol
                stopIteration = it;
                earlyStopped(fold) = true;
                fprintf('Early stopping at iteration %d (change %.3e).\n',it,lossChange);
                break
            end
        end
    end
    % Report optimization time without graphical rendering overhead.
    trainTime(fold) = max(toc(tTrain)-livePlotTime(fold),0);

    actualIterations(fold) = stopIteration;
    lossHistory = lossHistory(1:stopIteration);
    lossDataHistory = lossDataHistory(1:stopIteration);
    lossPhysHistory = lossPhysHistory(1:stopIteration);
    lossRegHistory = lossRegHistory(1:stopIteration);
    lossDataWeightedHistory = lossDataWeightedHistory(1:stopIteration);
    lossPhysWeightedHistory = lossPhysWeightedHistory(1:stopIteration);
    lossRegWeightedHistory = lossRegWeightedHistory(1:stopIteration);
    lossYuHistory = lossYuHistory(1:stopIteration);
    lossLiangHistory = lossLiangHistory(1:stopIteration);
    lossLDHistory = lossLDHistory(1:stopIteration);

    if saveVideo && ~isempty(videoWriter)
        close(videoWriter);
        fprintf('Fold %d video saved: %s\n',fold,videoFile);
    end

    predTr = predictReal(net,xData,muy,stdy);
    predVa = predictReal(net,xVal,muy,stdy);
    testPredByFold(:,fold) = predictReal(net,xTest,muy,stdy);
    trainMetrics(fold,:) = calcMetrics(ytr,predTr);
    valMetrics(fold,:) = calcMetrics(yva,predVa);

    oofPred(idxVa) = predVa;
    foldID(idxVa) = fold;
    foldTotalTime(fold) = toc(tFold);

    fprintf('Fold %d validation: R2 = %.4f | RMSE = %.3f kN | A20 = %.2f %%\n', ...
        fold,valMetrics(fold,1),valMetrics(fold,4),valMetrics(fold,6));
    fprintf('Fold %d training time = %.2f s (%.2f min)\n', ...
        fold,trainTime(fold),trainTime(fold)/60);
    if showLivePlot || saveVideo
        fprintf('Fold %d live-plot overhead = %.2f s\n',fold,livePlotTime(fold));
    end

    % Export detailed results separately for the current fold.
    foldExcel = fullfile(foldDir,sprintf('MINN_Results_Fold_%02d.xlsx',fold));
    FoldTrainPrediction = table(Xtr(:,1),Xtr(:,2),Xtr(:,3),Xtr(:,4), ...
        Xtr(:,5),ytr,predTr,predTr-ytr,predTr./ytr, ...
        'VariableNames',{'D','t','Fy','Fc','L_D','N_exp','N_pred', ...
        'Error','Npred_Nexp'});
    FoldValidationPrediction = table(Xva(:,1),Xva(:,2),Xva(:,3),Xva(:,4), ...
        Xva(:,5),yva,predVa,predVa-yva,predVa./yva, ...
        'VariableNames',{'D','t','Fy','Fc','L_D','N_exp','N_pred', ...
        'Error','Npred_Nexp'});
    FoldLossHistory = table((1:stopIteration)',lossHistory,lossDataHistory, ...
        lossPhysHistory,lossRegHistory,lossDataWeightedHistory, ...
        lossPhysWeightedHistory,lossRegWeightedHistory,lossYuHistory, ...
        lossLiangHistory,lossLDHistory, ...
        'VariableNames',{'Iteration','TotalLoss','DataLoss','MechanicsLoss', ...
        'RegularizationLoss','DataLoss_Weighted','MechanicsLoss_Weighted', ...
        'RegularizationLoss_Weighted','YuLoss_Raw','LiangLoss_Raw', ...
        'LDLoss_Raw'});
    FoldMetricTable = table(metricNames',trainMetrics(fold,:)', ...
        valMetrics(fold,:)', ...
        'VariableNames',{'Metric','Training','Validation'});
    writetable(FoldTrainPrediction,foldExcel,'Sheet','TRAINING_PREDICTION');
    writetable(FoldValidationPrediction,foldExcel,'Sheet','VALIDATION_PREDICTION');
    writetable(FoldMetricTable,foldExcel,'Sheet','METRICS');
    writetable(FoldLossHistory,foldExcel,'Sheet','LOSS_HISTORY');

    fLoss = figure('Visible','off');
    semilogy(1:stopIteration,lossHistory,'k','LineWidth',1.6); hold on;
    semilogy(1:stopIteration,lossDataWeightedHistory,'b','LineWidth',1.2);
    semilogy(1:stopIteration,lossPhysWeightedHistory,'r','LineWidth',1.2);
    semilogy(1:stopIteration,lossRegWeightedHistory,'m','LineWidth',1.2);
    hold off; grid on; xlabel('Iteration'); ylabel('Loss');
    legend('Total Loss','\lambda_d Data','\lambda_m Mechanics', ...
        '\lambda_r Reg','Location','best');
    title(sprintf('Model %d Training Loss',fold));
    exportgraphics(fLoss,fullfile(foldDir,'Final_Total_Loss.png'), ...
        'Resolution',1000);

    close(fLoss);

    fMechanics = figure('Visible','off');
    semilogy(1:stopIteration,lossYuHistory,'Color',[0.85 0.33 0.10], ...
        'LineWidth',1.3); hold on;
    semilogy(1:stopIteration,lossLiangHistory,'m','LineWidth',1.3);
    semilogy(1:stopIteration,lossLDHistory,'Color',[0.10 0.60 0.20], ...
        'LineWidth',1.3); hold off; grid on;
    xlabel('Iteration'); ylabel('Raw mechanics loss');
    legend('Yu et al.','Liang & Fragomeni','L/D constraint','Location','best');
    title(sprintf('Model %d Mechanics-Loss Breakdown',fold));
    exportgraphics(fMechanics,fullfile(foldDir, ...
        'Final_Mechanics_Loss.png'),'Resolution',1200);
    
    close(fMechanics);

    % Final three-panel plot matching the original program, but separated
    % by fold and using validation rather than the independent testing set.
    fFinal = figure('Visible','off','Position',[50 100 1800 560]);
    subplot(1,3,1)
    semilogy(1:stopIteration,lossHistory,'k','LineWidth',1.6); hold on;
    semilogy(1:stopIteration,lossDataWeightedHistory,'b','LineWidth',1.2);
    semilogy(1:stopIteration,lossPhysWeightedHistory,'r','LineWidth',1.2);
    semilogy(1:stopIteration,lossRegWeightedHistory,'m','LineWidth',1.2);
    hold off; grid on; xlabel('Iteration'); ylabel('Loss');
    legend('Total Loss','\lambda_d Data','\lambda_m Mechanics', ...
        '\lambda_r Reg','Location','best');
    title(sprintf('Model %d Weighted Loss',fold));

    subplot(1,3,2)
    semilogy(1:stopIteration,lossYuHistory,'Color',[0.85 0.33 0.10], ...
        'LineWidth',1.2); hold on;
    semilogy(1:stopIteration,lossLiangHistory,'m','LineWidth',1.2);
    semilogy(1:stopIteration,lossLDHistory,'g','LineWidth',1.2);
    hold off; grid on; xlabel('Iteration'); ylabel('Raw Mechanics Loss');
    legend('Yu et al.','Liang & Fragomeni','L/D constraint','Location','best');
    title(sprintf('Model %d Mechanics Loss',fold));

    subplot(1,3,3)
    scatter(ytr,predTr,22,'filled'); hold on;
    scatter(yva,predVa,28,'filled');
    mnFold=min([ytr;yva;predTr;predVa]); mxFold=max([ytr;yva;predTr;predVa]);
    plot([mnFold mxFold],[mnFold mxFold],'k--','LineWidth',1.2);
    hold off; grid on;
    xlabel('Experimental N_u (kN)'); ylabel('Predicted N_u (kN)');
    legend('Training','Validation','1:1 Line','Location','best');
    title(sprintf('Model %d | Train R^2=%.4f | Validation R^2=%.4f', ...
        fold,trainMetrics(fold,1),valMetrics(fold,1)));
    exportgraphics(fFinal,fullfile(foldDir,'Final_Training_Plot.png'), ...
        'Resolution',1000);
    close(fFinal);

    % Save every fold model because no single fold model represents all data.
    save(fullfile(foldDir,sprintf('MINN_Model_Fold_%02d.mat',fold)), ...
        'net','muX','stdX','muy','stdy','idxTr','idxVa','stopIteration', ...
        'lossHistory','lossDataHistory','lossPhysHistory','lossRegHistory', ...
        'lossDataWeightedHistory','lossPhysWeightedHistory', ...
        'lossRegWeightedHistory','lossYuHistory', ...
        'lossLiangHistory','lossLDHistory');
end

assert(all(isfinite(oofPred)),'Some samples did not receive an OOF prediction.');

%% 4) SUMMARY AND EXPORT
oofMetrics = calcMetrics(y,oofPred);
testPred = mean(testPredByFold,2);
testMetrics = calcMetrics(ytest,testPred);
meanTrain = mean(trainMetrics,1);
stdTrain = std(trainMetrics,0,1);
meanVal = mean(valMetrics,1);
stdVal = std(valMetrics,0,1);

FoldMetrics = table((1:K)',sum(training(cv,1)) .* ones(K,1), ...
    nan(K,1),actualIterations,earlyStopped,trainTime,trainTime./60, ...
    livePlotTime,foldTotalTime, ...
    'VariableNames',{'Fold','N_Training','N_Validation','ActualIterations', ...
    'EarlyStopped','TrainingTime_seconds','TrainingTime_minutes', ...
    'LivePlotTime_seconds','TotalFoldTime_seconds'});
for fold = 1:K
    FoldMetrics.N_Training(fold) = sum(training(cv,fold));
    FoldMetrics.N_Validation(fold) = sum(test(cv,fold));
end
for j = 1:numel(metricNames)
    FoldMetrics.(['Train_' metricNames{j}]) = trainMetrics(:,j);
    FoldMetrics.(['Validation_' metricNames{j}]) = valMetrics(:,j);
end

Summary = table(metricNames',meanTrain',stdTrain',meanVal',stdVal',oofMetrics', ...
    'VariableNames',{'Metric','Train_Mean','Train_SD','Validation_Mean', ...
    'Validation_SD','OOF_Training'});

OOFPredictions = table((1:n)',foldID,X(:,1),X(:,2),X(:,3),X(:,4),X(:,5), ...
    y,oofPred,oofPred-y,abs(oofPred-y),oofPred./y, ...
    'VariableNames',{'Sample_ID','Fold','D','t','Fy','Fc','L_D', ...
    'N_exp','N_pred_OOF','Error','Abs_Error','Npred_Nexp'});

TestPredictions = table((1:nTest)',Xtest(:,1),Xtest(:,2),Xtest(:,3), ...
    Xtest(:,4),Xtest(:,5),ytest,testPred,testPred-ytest,abs(testPred-ytest), ...
    testPred./ytest, ...
    'VariableNames',{'Sample_ID','D','t','Fy','Fc','L_D','N_exp', ...
    'N_pred_Ensemble','Error','Abs_Error','Npred_Nexp'});
for fold = 1:K
    TestPredictions.(sprintf('N_pred_Fold_%02d',fold)) = testPredByFold(:,fold);
end

TestMetrics = table(metricNames',testMetrics', ...
    'VariableNames',{'Metric','Independent_Testing'});
OOFTrainingMetrics = table(metricNames',oofMetrics', ...
    'VariableNames',{'Metric','OOF_Training'});
AllMetrics = table(metricNames',oofMetrics',testMetrics', ...
    'VariableNames',{'Metric','OOF_Training','Independent_Testing'});

Settings = table(K,fixedSeed,numHL,numNeuron,learnRate,numIterMax,delta, ...
    lambda_data_total,lambda_phys_total,lambda_yu,lambda_liang,lambda_ld, ...
    lambda_reg,checkEvery,lossTol,minIterCheck,showLivePlot,livePlotEvery, ...
    saveVideo, ...
    'VariableNames',{'K','FixedSeed','HiddenLayers','Neurons','LearningRate', ...
    'MaxIterations','Delta','LambdaData','LambdaPhysics','LambdaYu', ...
    'LambdaLiang','LambdaLD','LambdaReg','CheckEvery','LossTolerance', ...
    'MinIterationCheck','ShowLivePlot','LivePlotEvery','SaveVideo'});

excelFile = fullfile(outDir,'MINN_N_S_5Fold_Results.xlsx');
writetable(FoldMetrics,excelFile,'Sheet','FOLD_METRICS');
writetable(Summary,excelFile,'Sheet','SUMMARY_MEAN_SD');
writetable(OOFPredictions,excelFile,'Sheet','OOF_PREDICTIONS');
writetable(TestPredictions,excelFile,'Sheet','TESTING_PREDICTIONS');
writetable(OOFTrainingMetrics,excelFile,'Sheet','OOF_TRAINING_METRICS');
writetable(TestMetrics,excelFile,'Sheet','TESTING_METRICS');
writetable(AllMetrics,excelFile,'Sheet','ALL_METRICS');
writetable(Settings,excelFile,'Sheet','SETTINGS');

fParity = figure('Position',[100 100 760 620]);
gscatter(y,oofPred,foldID,lines(K),'.',18); hold on;
mn = min([y;oofPred]); mx = max([y;oofPred]);
plot([mn mx],[mn mx],'k--','LineWidth',1.4); hold off; grid on;
xlabel('Experimental N_u (kN)'); ylabel('OOF Predicted N_u (kN)');
title(sprintf('5-Fold OOF Parity | R^2 = %.4f | RMSE = %.2f kN', ...
    oofMetrics(1),oofMetrics(4)));
legend([compose('Fold %d',1:K),'1:1 Line'],'Location','best');
exportgraphics(fParity,fullfile(outDir,'OOF_Parity_Plot.png'),'Resolution',1000);
exportgraphics(fParity,fullfile(outDir,'OOF_Parity_Plot.tif'),'Resolution',1000);

fTest = figure('Position',[100 100 760 620]);
scatter(ytest,testPred,35,'filled'); hold on;
mn = min([ytest;testPred]); mx = max([ytest;testPred]);
plot([mn mx],[mn mx],'k--','LineWidth',1.4); hold off; grid on;
xlabel('Experimental N_u (kN)'); ylabel('Ensemble Predicted N_u (kN)');
title(sprintf('Independent Testing | R^2 = %.4f | RMSE = %.2f kN', ...
    testMetrics(1),testMetrics(4)));
legend('Testing data','1:1 Line','Location','best');
exportgraphics(fTest,fullfile(outDir,'Independent_Testing_Parity.png'),'Resolution',1000);
exportgraphics(fTest,fullfile(outDir,'Independent_Testing_Parity.tif'),'Resolution',1000);

fprintf('\n================ 5-FOLD VALIDATION SUMMARY ================\n');
fprintf('OOF Training R2   = %.4f\n',oofMetrics(1));
fprintf('OOF Training MAE  = %.3f kN\n',oofMetrics(2));
fprintf('OOF Training RMSE = %.3f kN\n',oofMetrics(4));
fprintf('OOF Training A20  = %.2f %%\n',oofMetrics(6));
fprintf('Validation R2     = %.4f +/- %.4f\n',meanVal(1),stdVal(1));
fprintf('Validation MAE    = %.3f +/- %.3f kN\n',meanVal(2),stdVal(2));
fprintf('Validation RMSE   = %.3f +/- %.3f kN\n',meanVal(4),stdVal(4));
fprintf('Validation A20    = %.2f +/- %.2f %%\n',meanVal(6),stdVal(6));
fprintf('\n------------- INDEPENDENT TESTING -------------\n');
fprintf('Testing R2     = %.4f\n',testMetrics(1));
fprintf('Testing MAE    = %.3f kN\n',testMetrics(2));
fprintf('Testing RMSE   = %.3f kN\n',testMetrics(4));
fprintf('Testing A20    = %.2f %%\n',testMetrics(6));
fprintf('Testing mean ratio = %.4f\n',testMetrics(7));
fprintf('\n---------------- TRAINING TIME ----------------\n');
fprintf('Fold training times (s): %s\n',mat2str(trainTime',4));
fprintf('Mean training time = %.2f +/- %.2f s/fold\n', ...
    mean(trainTime),std(trainTime));
fprintf('Mean training time = %.2f +/- %.2f min/fold\n', ...
    mean(trainTime)/60,std(trainTime)/60);
fprintf('Total training time for 5 folds = %.2f s (%.2f min, %.2f h)\n', ...
    sum(trainTime),sum(trainTime)/60,sum(trainTime)/3600);
if showLivePlot || saveVideo
    fprintf('Total live-plot overhead = %.2f s (excluded from training time)\n', ...
        sum(livePlotTime));
end
fprintf('Results: %s\n',excelFile);
fprintf('Total time: %.2f sec\n',toc(tTotal));

%% ======================== LOCAL FUNCTIONS ========================
function pred = predictReal(net,x,muy,stdy)
    p = forward(net,x,'Outputs','output');
    pred = gather(extractdata(p))' .* stdy + muy;
end

function m = calcMetrics(obs,pred)
    err = pred-obs;
    mae = mean(abs(err)); mse = mean(err.^2); rmse = sqrt(mse);
    den = sum((obs-mean(obs)).^2);
    if den>0, r2 = 1-sum(err.^2)/den; else, r2 = NaN; end
    valid = obs~=0;
    if any(valid), mape = mean(abs(err(valid)./obs(valid)))*100; else, mape = NaN; end
    ratio = pred./obs;
    a20 = mean(ratio>=0.8 & ratio<=1.2)*100;
    m = [r2,mae,mse,rmse,mape,a20,mean(ratio),std(ratio)];
end

function [loss,grad,lossData,lossPhys,lossReg,lossYu,lossLiang,lossLD] = ...
    lossFun(net,x,y,xp,minX,maxX,muX,stdX,muy,stdy, ...
    ld,lm,ly,ll,lLD,lr,d)
    yp = forward(net,x,'Outputs','output');
    lossData = mean((yp-y).^2,'all');
    Xreal = extractdata(xp)' .* stdX + muX;
    Xmin = minX.*stdX+muX; Xmax = maxX.*stdX+muX;
    ybase = forward(net,xp,'Outputs','output').*stdy+muy;
    lossYu = equationTrend(net,Xreal,ybase,Xmin,Xmax,muX,stdX,muy,stdy,'yu',d);
    lossLiang = equationTrend(net,Xreal,ybase,Xmin,Xmax,muX,stdX,muy,stdy,'liang',d);
    lossLD = ldConstraint(net,Xreal,ybase,Xmin,Xmax,muX,stdX,muy,stdy,d);
    lossPhys = ly.*lossYu + ll.*lossLiang + lLD.*lossLD;
    reg = dlarray(0); vals = net.Learnables.Value;
    for q=1:numel(vals), reg=reg+sum(vals{q}.^2,'all'); end
    lossReg = lr.*reg;
    loss = ld.*lossData + lm.*lossPhys + lossReg;
    grad = dlgradient(loss,net.Learnables);
end

function lossLD = ldConstraint(net,X,y0,Xmin,Xmax,muX,stdX,muy,stdy,d)
    Xp=X; Xp(:,5)=min(max(Xp(:,5).*(1+d),Xmin(5)),Xmax(5));
    xp=dlarray(((Xp-muX)./stdX)','CB');
    y1=forward(net,xp,'Outputs','output').*stdy+muy;
    scale=max(mean(abs(extractdata(y0))),eps);
    lossLD=mean(max(0,y1-y0).^2,'all')./(scale^2);
end

function lossTrend = equationTrend(net,X,y0,Xmin,Xmax,muX,stdX,muy,stdy,name,d)
    [N0,v0]=equationValue(X,name); lossTrend=dlarray(0); nt=0;
    for k=1:4
        for s=[-1,1]
            X1=X; X1(:,k)=X1(:,k).*(1+s*d);
            X1(:,k)=min(max(X1(:,k),Xmin(k)),Xmax(k));
            x1=dlarray(((X1-muX)./stdX)','CB');
            y1=forward(net,x1,'Outputs','output').*stdy+muy;
            [N1,v1]=equationValue(X1,name);
            valid=v0 & v1 & isfinite(N0) & isfinite(N1) & N0>0 & N1>0;
            if any(valid)
                dEq=dlarray((N1(valid)-N0(valid))','CB');
                dPr=y1(:,valid')-y0(:,valid');
                scale=max(mean(abs(N0(valid))),eps);
                lossTrend=lossTrend+mean(((dPr-dEq)./scale).^2,'all'); nt=nt+1;
            end
        end
    end
    if nt>0, lossTrend=lossTrend./nt; end
end

function [N,valid] = equationValue(X,name)
    if strcmpi(name,'yu')
        N=calcYu(X); valid=isfinite(N)&N>0;
    else
        [N,valid]=calcLiang(X); valid=valid&isfinite(N)&N>0;
    end
end

function N = calcYu(X)
    D=max(X(:,1),eps); t=max(X(:,2),eps); fy=max(X(:,3),eps); fc=max(X(:,4),eps);
    Di=max(D-2.*t,eps); As=pi/4.*(D.^2-Di.^2); Ac=pi/4.*Di.^2;
    valid=fc>0&fy>0&Ac>0&As>0; N=nan(size(D));
    fck=0.8.*fc(valid); Asc=As(valid)+Ac(valid); beta=As(valid)./Asc;
    xi=(As(valid).*fy(valid))./(Ac(valid).*fck); om=ones(size(xi));
    eta=(om.*xi)./((2.*om+0.05.*xi+(0.2.*fck./fy(valid)-0.05).*xi.*om).*(om+xi));
    fsc=(1+eta).*((1-beta).*fck+beta.*fy(valid)); N(valid)=fsc.*Asc./1000;
end

function [N,valid] = calcLiang(X)
    D=max(X(:,1),eps); t=max(X(:,2),eps); fy=max(X(:,3),eps); fc=max(X(:,4),eps);
    Di=max(D-2.*t,eps); As=pi/4.*(D.^2-Di.^2); Ac=pi/4.*Di.^2; Dt=D./t;
    valid=Dt>0&Dt<=150&fc>0&fy>0&t>0&Di>0&fc./fy>=0.04&fc./fy<=0.20;
    N=nan(size(D)); if ~any(valid), return; end
    Dv=D(valid); tv=t(valid); fyv=fy(valid); fcv=fc(valid); Div=Di(valid);
    Asv=As(valid); Acv=Ac(valid); Dtv=Dv./tv;
    gc=min(max(1.85.*Div.^(-0.135),0.85),1); gs=min(max(1.458.*Dtv.^(-0.1),0.9),1.1);
    frp=zeros(size(Dtv)); i1=Dtv<=47; i2=Dtv>47&Dtv<=150;
    if any(i1)
        r=fcv(i1)./fyv(i1); ve0=0.881e-6.*Dtv(i1).^3-2.58e-4.*Dtv(i1).^2+1.953e-2.*Dtv(i1)+0.4011;
        ve=0.2312+0.3582.*ve0-0.1524.*r+4.843.*ve0.*r-9.169.*r.^2;
        frp(i1)=max(0,0.7.*(ve-0.5).*(2.*tv(i1)./max(Dv(i1)-2.*tv(i1),eps)).*fyv(i1));
    end
    if any(i2), frp(i2)=max(0,(0.006241-0.0000357.*Dtv(i2)).*fyv(i2)); end
    N(valid)=((gc.*fcv+4.1.*frp).*Acv+gs.*fyv.*Asv)./1000;
end
