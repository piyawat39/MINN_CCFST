clear; clc; close all;
tTotal = tic;

%% ============================================================
% ANN: 5-FOLD CROSS-VALIDATION + INDEPENDENT TESTING
% 5-fold is performed on the original training set only.
% The original testing set remains outside every fold.
%% ============================================================

%% 1) SETTINGS
trainFile = 'N_S_Training_DATASET_75.xlsx';
testFile  = 'N_S_Testing_DATASET_25.xlsx';
K = 5;
fixedSeed = 1;

numHL = 2;
numNeuron = 121;
learnRate = 0.000322957;
maxIter = 100000;

lambda_data_total = 0.005;
lambda_reg = 1e-7;

checkEvery = 5000;
earlyStopTol = 1e-5;
patience = 1;
minIter = checkEvery;
printEvery = 1000;

showLivePlot = true;       % true/false
saveVideo = false;         % true/false (independent of showLivePlot)
livePlotEvery = 1000;

outDir = 'Results_ANN_N_S_5Fold';
if ~exist(outDir,'dir'), mkdir(outDir); end

rng(fixedSeed,'twister');
try, gpuDevice([]); catch, end

%% 2) LOAD TRAINING AND INDEPENDENT TESTING SETS
Tr = readtable(trainFile,'VariableNamingRule','preserve');
Te = readtable(testFile,'VariableNamingRule','preserve');
Tr.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Tr.Properties.VariableNames));
Te.Properties.VariableNames = matlab.lang.makeValidName(strtrim(Te.Properties.VariableNames));

X = [Tr.D Tr.t Tr.fy Tr.fc Tr.L_D]; y = Tr.N;
Xtest = [Te.D Te.t Te.fy Te.fc Te.L_D]; ytest = Te.N;
n = size(X,1); nTest = size(Xtest,1);

assert(all(isfinite(X),'all') && all(isfinite(y)), ...
    'Training data contain NaN or Inf.');
assert(all(isfinite(Xtest),'all') && all(isfinite(ytest)), ...
    'Testing data contain NaN or Inf.');

cv = cvpartition(n,'KFold',K);
metricNames = {'R2','MAE','MSE','RMSE','MAPE_percent','A20_percent', ...
    'Mean_Npred_Nexp','SD_Npred_Nexp'};
trainMetrics = nan(K,numel(metricNames));
valMetrics = nan(K,numel(metricNames));
actualIterations = nan(K,1); earlyStopped = false(K,1);
trainTime = nan(K,1); livePlotTime = zeros(K,1); foldTotalTime = nan(K,1);
oofPred = nan(n,1); foldID = nan(n,1);
testPredByFold = nan(nTest,K);

fprintf('\n=====================================================\n');
fprintf('ANN %d-FOLD CROSS-VALIDATION | Training N = %d\n',K,n);
fprintf('Independent testing N = %d (excluded from all folds)\n',nTest);
fprintf('Architecture = %d hidden layers x %d neurons\n',numHL,numNeuron);
fprintf('=====================================================\n');

%% 3) FIVE-FOLD LOOP
for fold = 1:K
    tFold = tic;
    foldDir = fullfile(outDir,sprintf('Fold_%02d',fold));
    if ~exist(foldDir,'dir'), mkdir(foldDir); end

    idxTr = training(cv,fold); idxVa = test(cv,fold);
    Xtr = X(idxTr,:); ytr = y(idxTr);
    Xva = X(idxVa,:); yva = y(idxVa);

    % Fold-specific normalization: no validation/testing leakage.
    muX = mean(Xtr,1); stdX = std(Xtr,0,1); stdX(stdX==0)=1;
    muy = mean(ytr); stdy = std(ytr); if stdy==0, stdy=1; end
    XnTr=(Xtr-muX)./stdX; XnVa=(Xva-muX)./stdX;
    XnTest=(Xtest-muX)./stdX; ynTr=(ytr-muy)./stdy;
    xTr=dlarray(XnTr','CB'); yTr=dlarray(ynTr','CB');
    xVa=dlarray(XnVa','CB'); xTest=dlarray(XnTest','CB');

    rng(fixedSeed+fold-1,'twister');
    layers = featureInputLayer(5,'Normalization','none','Name','input');
    for j=1:numHL
        layers=[layers
            fullyConnectedLayer(numNeuron,'Name',sprintf('fc%d',j))
            tanhLayer('Name',sprintf('tanh%d',j))]; %#ok<AGROW>
    end
    layers=[layers;fullyConnectedLayer(1,'Name','output')];
    net=dlnetwork(layerGraph(layers));

    trailingAvg=[]; trailingAvgSq=[];
    lossHistory=nan(maxIter,1); lossDataHistory=nan(maxIter,1);
    lossRegHistory=nan(maxIter,1); dataWeightedHistory=nan(maxIter,1);
    stopCounter=0; actualIter=maxIter;

    fprintf('\n---------------- FOLD %d/%d ----------------\n',fold,K);
    fprintf('Training = %d | Validation = %d | Seed = %d\n', ...
        sum(idxTr),sum(idxVa),fixedSeed+fold-1);

    if showLivePlot || saveVideo
        if showLivePlot, vis='on'; else, vis='off'; end
        figLive=figure('Name',sprintf('ANN Live Training - Fold %d',fold), ...
            'NumberTitle','off','Visible',vis,'Position',[50 100 1400 560]);
    else
        figLive=[];
    end
    if saveVideo
        videoFile=fullfile(foldDir,sprintf('Live_Training_Fold_%02d.mp4',fold));
        videoWriter=VideoWriter(videoFile,'MPEG-4');
        videoWriter.FrameRate=5; videoWriter.Quality=100; open(videoWriter);
    else
        videoWriter=[];
    end

    tTrain=tic;
    for it=1:maxIter
        [loss,grad,lossData,lossReg] = dlfeval(@lossFunANN,net,xTr,yTr, ...
            lambda_data_total,lambda_reg);
        [net,trailingAvg,trailingAvgSq]=adamupdate(net,grad, ...
            trailingAvg,trailingAvgSq,it,learnRate);
        lossHistory(it)=gather(extractdata(loss));
        lossDataHistory(it)=gather(extractdata(lossData));
        lossRegHistory(it)=gather(extractdata(lossReg));
        dataWeightedHistory(it)=lambda_data_total*lossDataHistory(it);

        if (showLivePlot || saveVideo) && isgraphics(figLive) && ...
                (it==1 || mod(it,livePlotEvery)==0 || it==maxIter)
            tPlot=tic;
            predTrLive=predictReal(net,xTr,muy,stdy);
            predVaLive=predictReal(net,xVa,muy,stdy);
            mTr=calcMetrics(ytr,predTrLive); mVa=calcMetrics(yva,predVaLive);
            figure(figLive); clf(figLive);
            tiledlayout(figLive,1,2,'TileSpacing','compact','Padding','compact');

            nexttile;
            semilogy(1:it,lossHistory(1:it),'k','LineWidth',1.6); hold on;
            semilogy(1:it,dataWeightedHistory(1:it),'b','LineWidth',1.3);
            semilogy(1:it,lossRegHistory(1:it),'m','LineWidth',1.3);
            hold off; grid on; xlabel('Iteration'); ylabel('Loss');
            legend('Total Loss','\lambda_d Data','\lambda_r Regularization', ...
                'Location','best');
            title(sprintf('Fold %d/%d ANN Training Loss',fold,K));

            nexttile;
            scatter(ytr,predTrLive,22,'filled'); hold on;
            scatter(yva,predVaLive,28,'filled');
            mn=min([ytr;yva;predTrLive;predVaLive]);
            mx=max([ytr;yva;predTrLive;predVaLive]);
            plot([mn mx],[mn mx],'k--','LineWidth',1.2); hold off; grid on;
            xlabel('Experimental N_u (kN)'); ylabel('Predicted N_u (kN)');
            legend('Training','Validation','1:1 Line','Location','best');
            title(sprintf('Train R^2=%.4f | Validation R^2=%.4f',mTr(1),mVa(1)));
            drawnow;

            if saveVideo && ~isempty(videoWriter)
                frame=getframe(figLive); img=frame.cdata;
                [hh,ww,~]=size(img);
                if mod(hh,2)~=0, img=img(1:end-1,:,:); end
                if mod(ww,2)~=0, img=img(:,1:end-1,:); end
                writeVideo(videoWriter,img);
            end
            livePlotTime(fold)=livePlotTime(fold)+toc(tPlot);
        end

        if mod(it,printEvery)==0
            fprintf('Fold %d | Iter %6d | Loss %.3e | Data %.3e | Reg %.3e\n', ...
                fold,it,lossHistory(it),lossDataHistory(it),lossRegHistory(it));
        end

        if it>=minIter && mod(it,checkEvery)==0
            lossDiff=abs(lossHistory(it)-lossHistory(it-checkEvery+1));
            if lossDiff<earlyStopTol, stopCounter=stopCounter+1;
            else, stopCounter=0; end
            fprintf('Early-stop check: change %.3e | counter %d/%d\n', ...
                lossDiff,stopCounter,patience);
            if stopCounter>=patience
                actualIter=it; earlyStopped(fold)=true;
                fprintf('Early stopping at iteration %d.\n',it); break
            end
        end
        actualIter=it;
    end
    trainTime(fold)=max(toc(tTrain)-livePlotTime(fold),0);
    actualIterations(fold)=actualIter;

    lossHistory=lossHistory(1:actualIter);
    lossDataHistory=lossDataHistory(1:actualIter);
    lossRegHistory=lossRegHistory(1:actualIter);
    dataWeightedHistory=dataWeightedHistory(1:actualIter);
    if saveVideo && ~isempty(videoWriter), close(videoWriter); end

    predTr=predictReal(net,xTr,muy,stdy);
    predVa=predictReal(net,xVa,muy,stdy);
    testPredByFold(:,fold)=predictReal(net,xTest,muy,stdy);
    trainMetrics(fold,:)=calcMetrics(ytr,predTr);
    valMetrics(fold,:)=calcMetrics(yva,predVa);
    oofPred(idxVa)=predVa; foldID(idxVa)=fold;
    foldTotalTime(fold)=toc(tFold);

    fprintf('Fold %d validation: R2=%.4f | RMSE=%.3f kN | A20=%.2f %%\n', ...
        fold,valMetrics(fold,1),valMetrics(fold,4),valMetrics(fold,6));
    fprintf('Fold %d training time: %.2f s (%.2f min)\n', ...
        fold,trainTime(fold),trainTime(fold)/60);

    % Separate Excel output for each fold.
    foldExcel=fullfile(foldDir,sprintf('ANN_Results_Fold_%02d.xlsx',fold));
    TrainPrediction=predictionTable(Xtr,ytr,predTr);
    ValidationPrediction=predictionTable(Xva,yva,predVa);
    FoldLoss=table((1:actualIter)',lossHistory,lossDataHistory, ...
        dataWeightedHistory,lossRegHistory, ...
        'VariableNames',{'Iteration','TotalLoss','DataLoss_Raw', ...
        'DataLoss_Weighted','RegularizationLoss'});
    writetable(TrainPrediction,foldExcel,'Sheet','TRAINING_PREDICTION');
    writetable(ValidationPrediction,foldExcel,'Sheet','VALIDATION_PREDICTION');
    writetable(FoldLoss,foldExcel,'Sheet','LOSS_HISTORY');
    % Write metrics in the same orientation as the MINN output.
    FoldMetric=table(metricNames',trainMetrics(fold,:)',valMetrics(fold,:)', ...
        'VariableNames',{'Metric','Training','Validation'});
    writetable(FoldMetric,foldExcel,'Sheet','METRICS');

    % Final loss plot is always saved, even when live plot/video are off.
    fLoss=figure('Visible','off','Position',[100 100 750 600]);
    semilogy(1:actualIter,lossHistory,'k','LineWidth',1.6); hold on;
    semilogy(1:actualIter,dataWeightedHistory,'b','LineWidth',1.3);
    semilogy(1:actualIter,lossRegHistory,'m','LineWidth',1.3);
    hold off; grid on; xlabel('Iteration'); ylabel('Loss');
    legend('Total Loss','\lambda_d Data','\lambda_r Regularization','Location','best');
    title(sprintf('Fold %d ANN Training Loss',fold));
    exportgraphics(fLoss,fullfile(foldDir,'Final_Total_Loss.png'),'Resolution',1000);
    exportgraphics(fLoss,fullfile(foldDir,'Final_Total_Loss.tif'),'Resolution',1000);
    close(fLoss);

    fFinal=figure('Visible','off','Position',[50 100 1400 560]);
    subplot(1,2,1)
    semilogy(1:actualIter,lossHistory,'k','LineWidth',1.6); hold on;
    semilogy(1:actualIter,dataWeightedHistory,'b','LineWidth',1.3);
    semilogy(1:actualIter,lossRegHistory,'m','LineWidth',1.3);
    hold off; grid on; xlabel('Iteration'); ylabel('Loss');
    legend('Total Loss','\lambda_d Data','\lambda_r Regularization','Location','best');
    title(sprintf('Fold %d ANN Training Loss',fold));
    subplot(1,2,2)
    scatter(ytr,predTr,22,'filled'); hold on; scatter(yva,predVa,28,'filled');
    mn=min([ytr;yva;predTr;predVa]); mx=max([ytr;yva;predTr;predVa]);
    plot([mn mx],[mn mx],'k--','LineWidth',1.2); hold off; grid on;
    xlabel('Experimental N_u (kN)'); ylabel('Predicted N_u (kN)');
    legend('Training','Validation','1:1 Line','Location','best');
    title(sprintf('Train R^2=%.4f | Validation R^2=%.4f', ...
        trainMetrics(fold,1),valMetrics(fold,1)));
    exportgraphics(fFinal,fullfile(foldDir,'Final_Training_Plot.png'),'Resolution',1000);
    close(fFinal);

    save(fullfile(foldDir,sprintf('ANN_Model_Fold_%02d.mat',fold)), ...
        'net','muX','stdX','muy','stdy','idxTr','idxVa','actualIter', ...
        'lossHistory','lossDataHistory','dataWeightedHistory','lossRegHistory');
end

assert(all(isfinite(oofPred)),'Some training samples lack OOF predictions.');

%% 4) SUMMARY AND INDEPENDENT TESTING
oofMetrics=calcMetrics(y,oofPred);
testPred=mean(testPredByFold,2);
testMetrics=calcMetrics(ytest,testPred);
meanTrain=mean(trainMetrics,1); stdTrain=std(trainMetrics,0,1);
meanVal=mean(valMetrics,1); stdVal=std(valMetrics,0,1);

FoldMetrics=table((1:K)',nan(K,1),nan(K,1),actualIterations, ...
    earlyStopped,trainTime,trainTime./60,livePlotTime,foldTotalTime, ...
    'VariableNames',{'Fold','N_Training','N_Validation','ActualIterations', ...
    'EarlyStopped','TrainingTime_seconds','TrainingTime_minutes', ...
    'LivePlotTime_seconds','TotalFoldTime_seconds'});
for fold=1:K
    FoldMetrics.N_Training(fold)=sum(training(cv,fold));
    FoldMetrics.N_Validation(fold)=sum(test(cv,fold));
end
for j=1:numel(metricNames)
    FoldMetrics.(['Train_' metricNames{j}])=trainMetrics(:,j);
    FoldMetrics.(['Validation_' metricNames{j}])=valMetrics(:,j);
end

Summary=table(metricNames',meanTrain',stdTrain',meanVal',stdVal',oofMetrics', ...
    'VariableNames',{'Metric','Train_Mean','Train_SD','Validation_Mean', ...
    'Validation_SD','Pooled_OOF'});
OOFPredictions=table((1:n)',foldID,X(:,1),X(:,2),X(:,3),X(:,4),X(:,5), ...
    y,oofPred,oofPred-y,abs(oofPred-y),oofPred./y, ...
    'VariableNames',{'Sample_ID','Fold','D','t','Fy','Fc','L_D','N_exp', ...
    'N_pred_OOF','Error','Abs_Error','Npred_Nexp'});
TestPredictions=table((1:nTest)',Xtest(:,1),Xtest(:,2),Xtest(:,3), ...
    Xtest(:,4),Xtest(:,5),ytest,testPred,testPred-ytest,abs(testPred-ytest), ...
    testPred./ytest, ...
    'VariableNames',{'Sample_ID','D','t','Fy','Fc','L_D','N_exp', ...
    'N_pred_Ensemble','Error','Abs_Error','Npred_Nexp'});
for fold=1:K
    TestPredictions.(sprintf('N_pred_Fold_%02d',fold))=testPredByFold(:,fold);
end
TestMetrics=table(metricNames',testMetrics', ...
    'VariableNames',{'Metric','Independent_Testing'});
OOFTrainingMetrics=table(metricNames',oofMetrics', ...
    'VariableNames',{'Metric','OOF_Training'});
AllMetrics=table(metricNames',oofMetrics',testMetrics', ...
    'VariableNames',{'Metric','OOF_Training','Independent_Testing'});
Settings=table(K,fixedSeed,numHL,numNeuron,learnRate,maxIter, ...
    lambda_data_total,lambda_reg,checkEvery,earlyStopTol,patience, ...
    showLivePlot,saveVideo,livePlotEvery, ...
    'VariableNames',{'K','FixedSeed','HiddenLayers','Neurons','LearningRate', ...
    'MaxIterations','LambdaData','LambdaReg','CheckEvery','EarlyStopTolerance', ...
    'Patience','ShowLivePlot','SaveVideo','LivePlotEvery'});

excelFile=fullfile(outDir,'ANN_N_S_5Fold_Results.xlsx');
writetable(FoldMetrics,excelFile,'Sheet','FOLD_METRICS');
writetable(Summary,excelFile,'Sheet','SUMMARY_MEAN_SD');
writetable(OOFPredictions,excelFile,'Sheet','OOF_PREDICTIONS');
writetable(TestPredictions,excelFile,'Sheet','TESTING_PREDICTIONS');
writetable(OOFTrainingMetrics,excelFile,'Sheet','OOF_TRAINING_METRICS');
writetable(TestMetrics,excelFile,'Sheet','TESTING_METRICS');
writetable(AllMetrics,excelFile,'Sheet','ALL_METRICS');
writetable(Settings,excelFile,'Sheet','SETTINGS');

fOOF=figure('Position',[100 100 760 620]);
gscatter(y,oofPred,foldID,lines(K),'.',18); hold on;
mn=min([y;oofPred]); mx=max([y;oofPred]); plot([mn mx],[mn mx],'k--','LineWidth',1.4);
hold off; grid on; xlabel('Experimental N_u (kN)'); ylabel('OOF Predicted N_u (kN)');
title(sprintf('ANN 5-Fold OOF | R^2=%.4f | RMSE=%.2f kN',oofMetrics(1),oofMetrics(4)));
exportgraphics(fOOF,fullfile(outDir,'OOF_Parity_Plot.png'),'Resolution',1000);

fTest=figure('Position',[100 100 760 620]);
scatter(ytest,testPred,35,'filled'); hold on;
mn=min([ytest;testPred]); mx=max([ytest;testPred]); plot([mn mx],[mn mx],'k--','LineWidth',1.4);
hold off; grid on; xlabel('Experimental N_u (kN)'); ylabel('Ensemble Predicted N_u (kN)');
title(sprintf('ANN Independent Testing | R^2=%.4f | RMSE=%.2f kN', ...
    testMetrics(1),testMetrics(4)));
exportgraphics(fTest,fullfile(outDir,'Independent_Testing_Parity.png'),'Resolution',1000);

fprintf('\n================ ANN 5-FOLD SUMMARY ================\n');
fprintf('OOF Training R2   = %.4f\n',oofMetrics(1));
fprintf('OOF Training RMSE = %.3f kN\n',oofMetrics(4));
fprintf('OOF Training A20  = %.2f %%\n',oofMetrics(6));
fprintf('Validation R2     = %.4f +/- %.4f\n',meanVal(1),stdVal(1));
fprintf('Validation RMSE   = %.3f +/- %.3f kN\n',meanVal(4),stdVal(4));
fprintf('Validation A20    = %.2f +/- %.2f %%\n',meanVal(6),stdVal(6));
fprintf('Testing R2        = %.4f\n',testMetrics(1));
fprintf('Testing RMSE      = %.3f kN\n',testMetrics(4));
fprintf('Testing A20       = %.2f %%\n',testMetrics(6));
fprintf('Mean train time = %.2f +/- %.2f min/fold\n', ...
    mean(trainTime)/60,std(trainTime)/60);
fprintf('Total train time = %.2f min (%.2f h)\n',sum(trainTime)/60,sum(trainTime)/3600);
fprintf('Results: %s\n',excelFile);
fprintf('Total program time: %.2f s\n',toc(tTotal));

%% LOCAL FUNCTIONS
function pred=predictReal(net,x,muy,stdy)
    p=forward(net,x,'Outputs','output');
    pred=gather(extractdata(p))'.*stdy+muy;
end

function m=calcMetrics(obs,pred)
    err=pred-obs; mae=mean(abs(err)); mse=mean(err.^2); rmse=sqrt(mse);
    den=sum((obs-mean(obs)).^2);
    if den>0, r2=1-sum(err.^2)/den; else, r2=NaN; end
    valid=obs~=0;
    if any(valid), mape=mean(abs(err(valid)./obs(valid)))*100; else, mape=NaN; end
    ratio=pred./obs; a20=mean(ratio>=0.8 & ratio<=1.2)*100;
    m=[r2,mae,mse,rmse,mape,a20,mean(ratio),std(ratio)];
end

function T=predictionTable(X,y,pred)
    T=table(X(:,1),X(:,2),X(:,3),X(:,4),X(:,5),y,pred,pred-y,pred./y, ...
        'VariableNames',{'D','t','Fy','Fc','L_D','N_exp','N_pred', ...
        'Error','Npred_Nexp'});
end

function [loss,grad,lossData,lossReg]=lossFunANN(net,x,y,lambdaData,lambdaReg)
    yp=forward(net,x,'Outputs','output');
    lossData=mean((yp-y).^2,'all');
    reg=dlarray(0); vals=net.Learnables.Value;
    for i=1:numel(vals), reg=reg+sum(vals{i}.^2,'all'); end
    lossReg=lambdaReg.*reg;
    loss=lambdaData.*lossData+lossReg;
    grad=dlgradient(loss,net.Learnables);
end
