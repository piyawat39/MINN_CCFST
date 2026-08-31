%% MINN MULTI-OBJECTIVE BAYESIAN OPTIMIZATION (ParEGO) + 5-FOLD CV
clear;clc;close all;tTotal=tic;

%% COMMAND WINDOW LOG
logDir='Results_MINN_MOBO_5Fold';
if ~exist(logDir,'dir'),mkdir(logDir);end
commandWindowFile=fullfile(logDir,'MINN_MOBO_5Fold_CommandWindow.txt');

% Start a fresh Command Window log for this run
if exist(commandWindowFile,'file'),delete(commandWindowFile);end
diary(commandWindowFile);
diary on;

% Ensure diary is closed even if the script stops with an error
diaryCleanup=onCleanup(@() diary('off'));

fprintf('====================================================\n');
fprintf('MINN MOBO 5-Fold run started: %s\n',datestr(now,'yyyy-mm-dd HH:MM:SS'));
fprintf('Command Window log: %s\n',commandWindowFile);
fprintf('====================================================\n');

%% 1) SETTINGS
trainFile='N_S_Training_DATASET_75.xlsx';
testFile ='N_S_Testing_DATASET_25.xlsx';
K=5;fixedSeed=1;
nInitial=12;             % initial space-filling evaluations
maxEvaluations=60;       % total configurations; total trainings = 60 x 5

% Search ranges (not grids)
lrRange=[1e-5,5e-4];
neuronRange=[2,128];
hiddenLayerRange=[2,5];
lambdaDataRange=[0.001,0.02];

cfg.lambdaPhys=1;cfg.lambdaYu=1;cfg.lambdaLiang=1;cfg.lambdaLD=1;
cfg.lambdaReg=1e-7;cfg.delta=0.01;cfg.numIter=100000;
cfg.checkEvery=5000;cfg.lossTol=1e-5;cfg.minIterCheck=10000;

outDir='Results_MINN_MOBO_5Fold';
if ~exist(outDir,'dir'),mkdir(outDir);end
checkpointFile=fullfile(outDir,'MOBO_CHECKPOINT.mat');
excelFile=fullfile(outDir,'MINN_MOBO_5Fold_Results.xlsx');
rng(fixedSeed,'twister');try,gpuDevice([]);catch,end

%% 2) LOAD DATA AND CREATE FIXED FOLDS
Tr=readtable(trainFile,'VariableNamingRule','preserve');
Te=readtable(testFile,'VariableNamingRule','preserve');
Tr.Properties.VariableNames=matlab.lang.makeValidName(strtrim(Tr.Properties.VariableNames));
Te.Properties.VariableNames=matlab.lang.makeValidName(strtrim(Te.Properties.VariableNames));
X=[Tr.D Tr.t Tr.fy Tr.fc Tr.L_D];y=Tr.N;
Xtest=[Te.D Te.t Te.fy Te.fc Te.L_D];ytest=Te.N;
cv=cvpartition(size(X,1),'KFold',K);

vars=[
    optimizableVariable('Log10LearningRate',log10(lrRange),'Type','real')
    optimizableVariable('NumNeurons',neuronRange,'Type','integer')
    optimizableVariable('NumHiddenLayers',hiddenLayerRange,'Type','integer')
    optimizableVariable('Log10LambdaData',log10(lambdaDataRange),'Type','real')];

%% 3) INITIAL LATIN-HYPERCUBE EVALUATIONS
U=lhsdesign(nInitial,4,'Criterion','maximin','Iterations',100);
XHistory=table;
XHistory.Log10LearningRate=log10(lrRange(1))+U(:,1)*diff(log10(lrRange));
XHistory.NumNeurons=round(neuronRange(1)+U(:,2)*diff(neuronRange));
XHistory.NumHiddenLayers=round(hiddenLayerRange(1)+U(:,3)*diff(hiddenLayerRange));
XHistory.Log10LambdaData=log10(lambdaDataRange(1))+U(:,4)*diff(log10(lambdaDataRange));
FHistory=nan(nInitial,2);Details=cell(maxEvaluations,1);

fprintf('\nMINN ParEGO | %d initial + %d Bayesian = %d configurations\n', ...
    nInitial,maxEvaluations-nInitial,maxEvaluations);
fprintf('Total planned trainings = %d x %d = %d\n',maxEvaluations,K,maxEvaluations*K);

for e=1:nInitial
    [FHistory(e,:),Details{e}]=evaluateMINN5Fold(XHistory(e,:),X,y,cv,cfg,fixedSeed,e);
    printEvaluation(e,XHistory(e,:),FHistory(e,:),Details{e});
    save(checkpointFile,'XHistory','FHistory','Details','e','cfg','cv');
end

%% 4) ParEGO SEQUENTIAL BAYESIAN OPTIMIZATION
global PAREGO_CONTEXT PAREGO_LAST_X PAREGO_LAST_F PAREGO_LAST_DETAIL
PAREGO_CONTEXT=struct('X',X,'y',y,'cv',cv,'cfg',cfg,'seed',fixedSeed, ...
    'evaluation',nInitial,'fmin',[],'frange',[],'weight',[]);

for e=nInitial+1:maxEvaluations
    % Random weight explores different regions of the Pareto front.
    w=rand(1,2);w=w./sum(w);
    fmin=min(FHistory,[],1);frange=max(FHistory,[],1)-fmin;
    frange(frange<eps)=1;
    scalarHistory=paregoScalar(FHistory,fmin,frange,w);
    PAREGO_CONTEXT.evaluation=e;
    PAREGO_CONTEXT.fmin=fmin;PAREGO_CONTEXT.frange=frange;
    PAREGO_CONTEXT.weight=w;
    PAREGO_LAST_X=[];PAREGO_LAST_F=[];PAREGO_LAST_DETAIL=[];

    % InitialX/InitialObjective allow the surrogate to reuse all evaluations.
    bayesopt(@paregoObjective,vars, ...
        'InitialX',XHistory,'InitialObjective',scalarHistory, ...
        'MaxObjectiveEvaluations',height(XHistory)+1, ...
        'IsObjectiveDeterministic',true,'UseParallel',false, ...
        'Verbose',0,'PlotFcn',[],'AcquisitionFunctionName','expected-improvement-plus');

    if isempty(PAREGO_LAST_X)
        error('Bayesian optimizer did not generate a new evaluation at step %d.',e);
    end
    XHistory=[XHistory;PAREGO_LAST_X]; %#ok<AGROW>
    FHistory=[FHistory;PAREGO_LAST_F]; %#ok<AGROW>
    Details{e}=PAREGO_LAST_DETAIL;
    printEvaluation(e,PAREGO_LAST_X,PAREGO_LAST_F,PAREGO_LAST_DETAIL);
    save(checkpointFile,'XHistory','FHistory','Details','e','cfg','cv');
end

%% 5) PARETO FRONT AND BEST COMPROMISE
paretoMask=paretoMask2D(FHistory);
rmse=FHistory(paretoMask,1);mechLoss=FHistory(paretoMask,2);
z1=(rmse-min(rmse))./max(max(rmse)-min(rmse),eps);
z2=(mechLoss-min(mechLoss))./max(max(mechLoss)-min(mechLoss),eps);
[~,localBest]=min(sqrt(z1.^2+z2.^2));
paretoIdx=find(paretoMask);bestIdx=paretoIdx(localBest);
bestX=XHistory(bestIdx,:);bestDetail=Details{bestIdx};

%% 6) INDEPENDENT TESTING BY FIVE-MODEL ENSEMBLE
testPredByFold=nan(size(Xtest,1),K);bestModels=cell(K,1);bestNorm=cell(K,1);
for fold=1:K
    idxTr=training(cv,fold);
    [~,net,normInfo]=trainMINN(X(idxTr,:),y(idxTr),bestX,cfg, ...
        fixedSeed+fold-1);
    testPredByFold(:,fold)=predictReal(net,Xtest,normInfo);
    bestModels{fold}=net;bestNorm{fold}=normInfo;
end
testPred=mean(testPredByFold,2);testMetrics=metrics(ytest,testPred);

%% 7) EXPORT
Results=XHistory;
Results.LearningRate=10.^Results.Log10LearningRate;
Results.LambdaData=10.^Results.Log10LambdaData;
Results.Validation_RMSE_Mean=FHistory(:,1);
Results.MechanicsConsistency_Mean=100-FHistory(:,2);
Results.IsPareto=paretoMask;Results.IsBestCompromise=false(height(Results),1);
Results.IsBestCompromise(bestIdx)=true;
Results=movevars(Results,{'LearningRate','LambdaData'},'After','Log10LambdaData');
ParetoResults=Results(paretoMask,:);
TestingPrediction=table((1:numel(ytest))',Xtest(:,1),Xtest(:,2),Xtest(:,3), ...
    Xtest(:,4),Xtest(:,5),ytest,testPred,testPred-ytest,testPred./ytest, ...
    'VariableNames',{'Sample_ID','D','t','Fy','Fc','L_D','N_exp', ...
    'N_pred_Ensemble','Error','Npred_Nexp'});
for fold=1:K,TestingPrediction.(sprintf('N_pred_Fold_%02d',fold))=testPredByFold(:,fold);end
TestMetrics=struct2table(testMetrics);
Settings=table(K,fixedSeed,nInitial,maxEvaluations,maxEvaluations*K,cfg.numIter, ...
    cfg.lambdaPhys,cfg.lambdaReg,cfg.delta, ...
    'VariableNames',{'K','FixedSeed','InitialEvaluations','MaxEvaluations', ...
    'TotalTrainings','MaxIterations','LambdaPhysics','LambdaReg','Delta'});

% ---- Fold-by-fold details for every Bayesian evaluation ----
EvalFoldDetails=table;
for ee=1:maxEvaluations
    if ee<=numel(Details) && ~isempty(Details{ee})
        d=Details{ee};
        nFold=numel(d.Fold_RMSE);
        tmp=table( ...
            repmat(ee,nFold,1), ...
            (1:nFold)', ...
            repmat(XHistory.NumHiddenLayers(ee),nFold,1), ...
            repmat(XHistory.NumNeurons(ee),nFold,1), ...
            repmat(10^XHistory.Log10LearningRate(ee),nFold,1), ...
            repmat(10^XHistory.Log10LambdaData(ee),nFold,1), ...
            d.Fold_RMSE(:), ...
            d.Fold_Mechanics(:), ...
            'VariableNames',{'Evaluation','Fold','HiddenLayers','NumNeurons', ...
            'LearningRate','LambdaData','Validation_RMSE','MechanicsConsistency'});
        EvalFoldDetails=[EvalFoldDetails;tmp]; %#ok<AGROW>
    end
end

% ---- Compact best-compromise summary ----
BestCompromiseSummary=table( ...
    bestIdx, ...
    bestX.NumHiddenLayers, ...
    bestX.NumNeurons, ...
    10^bestX.Log10LearningRate, ...
    10^bestX.Log10LambdaData, ...
    FHistory(bestIdx,1), ...
    100-FHistory(bestIdx,2), ...
    testMetrics.MAE, ...
    testMetrics.MSE, ...
    testMetrics.RMSE, ...
    testMetrics.R2, ...
    testMetrics.A20, ...
    testMetrics.MeanRatio, ...
    testMetrics.SDRatio, ...
    'VariableNames',{'BestEvaluation','HiddenLayers','NumNeurons', ...
    'LearningRate','LambdaData','Validation_RMSE_Mean', ...
    'MechanicsConsistency_Mean','Testing_MAE','Testing_MSE', ...
    'Testing_RMSE','Testing_R2','Testing_A20','Testing_MeanRatio', ...
    'Testing_SDRatio'});

writetable(Results,excelFile,'Sheet','ALL_EVALUATIONS');
writetable(ParetoResults,excelFile,'Sheet','PARETO_FRONT');
writetable(Results(bestIdx,:),excelFile,'Sheet','BEST_COMPROMISE');
writetable(BestCompromiseSummary,excelFile,'Sheet','BEST_SUMMARY');
writetable(EvalFoldDetails,excelFile,'Sheet','FOLD_DETAILS');
writetable(TestingPrediction,excelFile,'Sheet','INDEPENDENT_TESTING');
writetable(TestMetrics,excelFile,'Sheet','TESTING_METRICS');
writetable(Settings,excelFile,'Sheet','SETTINGS');

f=figure('Color','w','Position',[100 100 850 650]);
scatter(FHistory(:,1),100-FHistory(:,2),35,[.65 .65 .65],'filled');hold on;
scatter(rmse,100-mechLoss,65,'r','filled');
plot(FHistory(bestIdx,1),100-FHistory(bestIdx,2),'kp','MarkerSize',16, ...
    'MarkerFaceColor','y');grid on;
xlabel('Mean 5-fold Validation RMSE (kN)');ylabel('Mean Mechanics Consistency (%)');
legend('All evaluations','Pareto front','Best compromise','Location','best');
title('MINN Multi-objective Bayesian Optimization');
exportgraphics(f,fullfile(outDir,'MINN_MOBO_Pareto_Front.png'),'Resolution',1200);
exportgraphics(f,fullfile(outDir,'MINN_MOBO_Pareto_Front.tif'),'Resolution',1200);

save(fullfile(outDir,'MINN_MOBO_5Fold_Results.mat'),'Results','ParetoResults', ...
    'bestIdx','bestX','bestDetail','bestModels','bestNorm','testPred', ...
    'testPredByFold','testMetrics','cfg','cv');

fprintf('\n================ BEST COMPROMISE ================\n');
fprintf('HL=%d | Neurons=%d | LR=%.3e | lambda_d=%.4g\n', ...
    bestX.NumHiddenLayers,bestX.NumNeurons,10^bestX.Log10LearningRate, ...
    10^bestX.Log10LambdaData);
fprintf('Validation RMSE=%.3f kN | Mechanics=%.2f %%\n', ...
    FHistory(bestIdx,1),100-FHistory(bestIdx,2));
fprintf('Independent testing RMSE=%.3f kN | R2=%.4f | A20=%.2f %%\n', ...
    testMetrics.RMSE,testMetrics.R2,testMetrics.A20);
fprintf('Total time=%.2f h | Results: %s\n',toc(tTotal)/3600,outDir);
fprintf('Run completed: %s\n',datestr(now,'yyyy-mm-dd HH:MM:SS'));
fprintf('Command Window saved to: %s\n',commandWindowFile);
fprintf('Excel results saved to: %s\n',excelFile);
diary off;

% Export Command Window text into Excel as an additional sheet
if exist(commandWindowFile,'file')
    fid=fopen(commandWindowFile,'r');
    if fid~=-1
        C=textscan(fid,'%s','Delimiter','\n','Whitespace','');
        fclose(fid);
        CommandWindowLog=table(C{1},'VariableNames',{'CommandWindow'});
        writetable(CommandWindowLog,excelFile,'Sheet','COMMAND_WINDOW_LOG');
    end
end

%% ================= LOCAL FUNCTIONS =================
function scalar=paregoObjective(x)
global PAREGO_CONTEXT PAREGO_LAST_X PAREGO_LAST_F PAREGO_LAST_DETAIL
[f,d]=evaluateMINN5Fold(x,PAREGO_CONTEXT.X,PAREGO_CONTEXT.y, ...
    PAREGO_CONTEXT.cv,PAREGO_CONTEXT.cfg,PAREGO_CONTEXT.seed, ...
    PAREGO_CONTEXT.evaluation);
PAREGO_LAST_X=x;PAREGO_LAST_F=f;PAREGO_LAST_DETAIL=d;
scalar=paregoScalar(f,PAREGO_CONTEXT.fmin,PAREGO_CONTEXT.frange, ...
    PAREGO_CONTEXT.weight);
end

function s=paregoScalar(F,fmin,frange,w)
Z=(F-fmin)./frange;s=max(Z.*w,[],2)+0.05*sum(Z.*w,2);
end

function [f,d]=evaluateMINN5Fold(x,X,y,cv,cfg,seed,evaluation)
K=cv.NumTestSets;vm=nan(K,1);mc=nan(K,1);times=nan(K,1);iters=nan(K,1);
for fold=1:K
    tr=training(cv,fold);va=test(cv,fold);t=tic;
    [info,net,normInfo]=trainMINN(X(tr,:),y(tr),x,cfg,seed+fold-1);
    pred=predictReal(net,X(va,:),normInfo);m=metrics(y(va),pred);
    vm(fold)=m.RMSE;mc(fold)=mechanicsConsistency(net,X(va,:),normInfo,cfg.delta);
    times(fold)=toc(t);iters(fold)=info.Iterations;
end
f=[mean(vm),100-mean(mc)];
d=struct('Evaluation',evaluation,'Fold_RMSE',vm,'Fold_Mechanics',mc, ...
    'RMSE_Mean',mean(vm),'RMSE_SD',std(vm),'Mechanics_Mean',mean(mc), ...
    'Mechanics_SD',std(mc),'TrainingTime_Total',sum(times), ...
    'Iterations_Mean',mean(iters));
end

function [info,net,normInfo]=trainMINN(X,y,x,cfg,seed)
muX=mean(X,1);stdX=std(X,0,1);stdX(stdX==0)=1;
muy=mean(y);stdy=std(y);if stdy==0,stdy=1;end
Xn=(X-muX)./stdX;yn=(y-muy)./stdy;
xd=dlarray(Xn','CB');yd=dlarray(yn','CB');xp=xd;
minX=min(Xn,[],1);maxX=max(Xn,[],1);
rng(seed,'twister');layers=featureInputLayer(5,'Normalization','none','Name','input');
for L=1:x.NumHiddenLayers
    layers=[layers;fullyConnectedLayer(x.NumNeurons,'Name',sprintf('fc%d',L)); ...
        tanhLayer('Name',sprintf('tanh%d',L))]; %#ok<AGROW>
end
layers=[layers;fullyConnectedLayer(1,'Name','output')];net=dlnetwork(layerGraph(layers));
avg=[];avgSq=[];prev=NaN;stop=cfg.numIter;lambdaData=10^x.Log10LambdaData;
for it=1:cfg.numIter
    [loss,grad]=dlfeval(@lossFun,net,xd,yd,xp,minX,maxX,muX,stdX,muy,stdy, ...
        lambdaData,cfg);
    [net,avg,avgSq]=adamupdate(net,grad,avg,avgSq,it,10^x.Log10LearningRate);
    if mod(it,cfg.checkEvery)==0
        now=double(gather(extractdata(loss)));
        if it>=cfg.minIterCheck && isfinite(prev) && abs(now-prev)<cfg.lossTol
            stop=it;break
        end
        prev=now;
    end
end
normInfo=struct('muX',muX,'stdX',stdX,'muy',muy,'stdy',stdy, ...
    'domainMin',min(X,[],1),'domainMax',max(X,[],1));
info=struct('Iterations',stop);
end

function [loss,grad]=lossFun(net,x,y,xp,minX,maxX,muX,stdX,muy,stdy,ld,cfg)
yp=forward(net,x);data=mean((yp-y).^2,'all');
X=extractdata(xp)'.*stdX+muX;Xmin=minX.*stdX+muX;Xmax=maxX.*stdX+muX;
y0=forward(net,xp).*stdy+muy;
yu=trendLoss(net,X,y0,Xmin,Xmax,muX,stdX,muy,stdy,'yu',cfg.delta);
li=trendLoss(net,X,y0,Xmin,Xmax,muX,stdX,muy,stdy,'liang',cfg.delta);
lld=ldLoss(net,X,y0,Xmin,Xmax,muX,stdX,muy,stdy,cfg.delta);
reg=dlarray(0);vals=net.Learnables.Value;
for q=1:numel(vals),reg=reg+sum(vals{q}.^2,'all');end
loss=ld*data+cfg.lambdaPhys*(cfg.lambdaYu*yu+cfg.lambdaLiang*li+cfg.lambdaLD*lld)+cfg.lambdaReg*reg;
grad=dlgradient(loss,net.Learnables);
end

function L=ldLoss(net,X,y0,Xmin,Xmax,muX,stdX,muy,stdy,d)
Xp=X;Xp(:,5)=min(max(Xp(:,5).*(1+d),Xmin(5)),Xmax(5));
yp=forward(net,dlarray(((Xp-muX)./stdX)','CB')).*stdy+muy;
sc=max(mean(abs(extractdata(y0))),eps);L=mean(max(0,yp-y0).^2,'all')/sc^2;
end

function L=trendLoss(net,X,y0,Xmin,Xmax,muX,stdX,muy,stdy,name,d)
L=dlarray(0);n=0;[N0,v0]=eqValue(X,name);
for k=1:4
 for s=[-1 1]
  Xp=X;Xp(:,k)=min(max(Xp(:,k).*(1+s*d),Xmin(k)),Xmax(k));
  yp=forward(net,dlarray(((Xp-muX)./stdX)','CB')).*stdy+muy;
  [N1,v1]=eqValue(Xp,name);v=v0&v1&isfinite(N0)&isfinite(N1)&N0>0&N1>0;
  if any(v),de=dlarray((N1(v)-N0(v))','CB');dp=yp(:,v')-y0(:,v');
   sc=max(mean(abs(N0(v))),eps);L=L+mean(((dp-de)/sc).^2,'all');n=n+1;end
 end
end
if n>0,L=L/n;end
end

function p=predictReal(net,X,n)
p=gather(extractdata(forward(net,dlarray(((X-n.muX)./n.stdX)','CB'))))';p=p*n.stdy+n.muy;
end

function pct=mechanicsConsistency(net,X,n,d)
y0=predictReal(net,X,n);yuAgree=[];liAgree=[];
[yu0,yv0]=eqValue(X,'yu');[li0,lv0]=eqValue(X,'liang');
for k=1:4
 Xp=X;Xp(:,k)=min(max(Xp(:,k).*(1+d),n.domainMin(k)),n.domainMax(k));
 dp=predictReal(net,Xp,n)-y0;
 [yu1,yv1]=eqValue(Xp,'yu');vy=yv0&yv1;
 [li1,lv1]=eqValue(Xp,'liang');vl=lv0&lv1;
 yuAgree=[yuAgree;double(sign(dp(vy))==sign(yu1(vy)-yu0(vy)))]; %#ok<AGROW>
 liAgree=[liAgree;double(sign(dp(vl))==sign(li1(vl)-li0(vl)))]; %#ok<AGROW>
end
Xld=X;Xld(:,5)=min(max(Xld(:,5).*(1+d),n.domainMin(5)),n.domainMax(5));
ld=100*mean(predictReal(net,Xld,n)<=y0+1e-9);
pct=mean([100*mean(yuAgree),100*mean(liAgree),ld],'omitnan');
end

function m=metrics(y,p)
e=p-y;r=p./y;m=struct('MAE',mean(abs(e)),'MSE',mean(e.^2), ...
 'RMSE',sqrt(mean(e.^2)),'R2',1-sum(e.^2)/sum((y-mean(y)).^2), ...
 'A20',100*mean(r>=.8&r<=1.2),'MeanRatio',mean(r),'SDRatio',std(r));
end

function mask=paretoMask2D(F)
n=size(F,1);mask=true(n,1);
for i=1:n
 for j=1:n
  if j~=i && all(F(j,:)<=F(i,:)) && any(F(j,:)<F(i,:)),mask(i)=false;break;end
 end
end
end

function printEvaluation(e,x,f,d)
fprintf('\nEval %d | HL=%d N=%d LR=%.2e lambda_d=%.4g\n',e,x.NumHiddenLayers, ...
 x.NumNeurons,10^x.Log10LearningRate,10^x.Log10LambdaData);
fprintf(' Validation RMSE %.3f +/- %.3f | Mechanics %.2f +/- %.2f %% | %.2f min\n', ...
 f(1),d.RMSE_SD,100-f(2),d.Mechanics_SD,d.TrainingTime_Total/60);
end

function [N,v]=eqValue(X,name)
D=max(X(:,1),eps);t=max(X(:,2),eps);fy=max(X(:,3),eps);fc=max(X(:,4),eps);
Di=max(D-2*t,eps);As=pi/4*(D.^2-Di.^2);Ac=pi/4*Di.^2;
if strcmpi(name,'yu')
 v=fc>0&fy>0&Ac>0&As>0;N=nan(size(D));fck=.8*fc(v);Asc=As(v)+Ac(v);
 b=As(v)./Asc;xi=As(v).*fy(v)./(Ac(v).*fck);o=ones(size(xi));
 eta=(o.*xi)./((2*o+.05*xi+(.2*fck./fy(v)-.05).*xi.*o).*(o+xi));
 N(v)=(1+eta).*((1-b).*fck+b.*fy(v)).*Asc/1000;
else
 Dt=D./t;r=fc./fy;v=Dt>0&Dt<=150&r>=.04&r<=.20;N=nan(size(D));
 if any(v),Dv=D(v);tv=t(v);fyv=fy(v);fcv=fc(v);Div=Di(v);Dtv=Dv./tv;
  gc=min(max(1.85*Div.^(-.135),.85),1);gs=min(max(1.458*Dtv.^(-.1),.9),1.1);
  fr=zeros(size(Dtv));i1=Dtv<=47;i2=~i1;
  q=fcv(i1)./fyv(i1);ve0=.881e-6*Dtv(i1).^3-2.58e-4*Dtv(i1).^2+1.953e-2*Dtv(i1)+.4011;
  ve=.2312+.3582*ve0-.1524*q+4.843*ve0.*q-9.169*q.^2;
  fr(i1)=max(.7*(ve-.5).*(2*tv(i1)./max(Dv(i1)-2*tv(i1),eps)).*fyv(i1),0);
  fr(i2)=max((.006241-.0000357*Dtv(i2)).*fyv(i2),0);
  N(v)=((gc.*fcv+4.1*fr).*Ac(v)+gs.*fyv.*As(v))/1000;end
end
v=v&isfinite(N)&N>0;
end
