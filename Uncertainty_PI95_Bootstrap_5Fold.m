%% ========================================================================
% Uncertainty_PI95_Bootstrap_5Fold.m
% Residual-calibrated PI95 + bootstrap 95% CIs for completed 5-fold MINN
% and ANN results. No re-training is required.
% ========================================================================
clear; clc; close all;

alpha = 0.05;                % PI95 target level
B = 10000;                   % bootstrap replications
rng(20260830,'twister');     % reproducible bootstrap resampling
models = {'MINN','ANN'};
modelColor = [0.00 0.45 0.74; 0.85 0.33 0.10];
testColor = [0.00 0.45 0.74];       % blue: independent testing
externalColor = [0.90 0.36 0.05];  % orange: external validation

outDir = uigetdir(pwd,'Select folder for uncertainty-analysis outputs');
if isequal(outDir,0), error('No output folder selected.'); end

PIrows = table(); CIrows = table(); Result = struct([]);
for m = 1:numel(models)
    name = models{m};
    resultDir = uigetdir(pwd,sprintf('Select Results_%s_N_S_5Fold folder',name));
    if isequal(resultDir,0), error('%s results folder was not selected.',name); end
    resultFile = chooseResultsWorkbook(resultDir,name);

    Toof = readtable(resultFile,'Sheet','OOF_PREDICTIONS','VariableNamingRule','preserve');
    Ttest = readtable(resultFile,'Sheet','TESTING_PREDICTIONS','VariableNamingRule','preserve');
    yOof = Toof.N_exp; pOof = Toof.N_pred_OOF;
    y = Ttest.N_exp; p = Ttest.N_pred_Ensemble;
    validOof = isfinite(yOof) & isfinite(pOof);
    validTest = isfinite(y) & isfinite(p) & y>0;
    yOof = yOof(validOof); pOof = pOof(validOof);
    y = y(validTest); p = p(validTest);

    % Finite-sample corrected residual-calibrated PI95.
    r = sort(abs(yOof-pOof)); nOof = numel(r);
    q95 = r(min(nOof,ceil((nOof+1)*(1-alpha))));
    lo = p-q95; hi = p+q95; width = hi-lo;
    covered = y>=lo & y<=hi;
    PIrows = [PIrows; table(string(name),nOof,numel(y),100*(1-alpha),q95, ...
        100*mean(covered),mean(width),median(width), ...
        'VariableNames',{'Model','N_OOF','N_Test','Nominal_PI_percent', ...
        'Residual_Radius_kN','Empirical_Coverage_percent', ...
        'Mean_PI_Width_kN','Median_PI_Width_kN'})]; %#ok<AGROW>
    Tpi = table(y,p,lo,hi,width,covered, ...
        'VariableNames',{'N_exp_kN','N_pred_kN','PI95_Lower_kN','PI95_Upper_kN', ...
        'PI95_Width_kN','Covered_by_PI95'});

    % Paired non-parametric bootstrap of testing metrics.
    n = numel(y); metricBoot = nan(B,3);
    for b = 1:B
        idx = randi(n,n,1); e = p(idx)-y(idx);
        metricBoot(b,1) = mean(p(idx)./y(idx));
        metricBoot(b,2) = mean(abs(e));
        metricBoot(b,3) = sqrt(mean(e.^2));
    end
    metricPoint = [mean(p./y), mean(abs(p-y)), sqrt(mean((p-y).^2))];
    metricNames = {'Mean prediction ratio','MAE (kN)','RMSE (kN)'};
    for j = 1:3
        ci = prctile(metricBoot(:,j),[2.5 97.5]);
        CIrows = [CIrows; table(string(name),string(metricNames{j}),metricPoint(j), ...
            ci(1),ci(2),B, ...
            'VariableNames',{'Model','Metric','Estimate','CI95_Lower','CI95_Upper','Bootstrap_Replications'})]; %#ok<AGROW>
    end
    Result(m).name = name; Result(m).y = y; Result(m).p = p; Result(m).lo = lo;
    Result(m).hi = hi; Result(m).covered = covered; Result(m).Tpi = Tpi;
    Result(m).metricBoot = metricBoot; Result(m).metricPoint = metricPoint;
    Result(m).q95 = q95;
end

excelFile = fullfile(outDir,'Prediction_Uncertainty_PI95_Bootstrap.xlsx');
writetable(PIrows,excelFile,'Sheet','PI95_SUMMARY');
writetable(CIrows,excelFile,'Sheet','BOOTSTRAP_CI');
for m = 1:numel(Result)
    writetable(Result(m).Tpi,excelFile,'Sheet',['PI95_TESTING_' Result(m).name]);
end

%% Paired bootstrap comparison: the same resampled test specimens are used
% for MINN and ANN in every repetition. Negative Delta means MINN is better.
if numel(Result)~=2 || numel(Result(1).y)~=numel(Result(2).y) || ...
        any(abs(Result(1).y-Result(2).y)>1e-9)
    error('Paired bootstrap requires MINN and ANN predictions for the same testing specimens in the same order.');
end
yPair = Result(1).y; pMINN = Result(1).p; pANN = Result(2).p; nPair = numel(yPair);
deltaBoot = nan(B,3);
for b = 1:B
    idx = randi(nPair,nPair,1);
    ratioMINN = mean(pMINN(idx)./yPair(idx));
    ratioANN  = mean(pANN(idx)./yPair(idx));
    deltaBoot(b,1) = abs(ratioMINN-1)-abs(ratioANN-1); % bias magnitude
    deltaBoot(b,2) = mean(abs(pMINN(idx)-yPair(idx))) - mean(abs(pANN(idx)-yPair(idx)));
    deltaBoot(b,3) = sqrt(mean((pMINN(idx)-yPair(idx)).^2)) - ...
                     sqrt(mean((pANN(idx)-yPair(idx)).^2));
end
deltaNames = {'Delta |mean ratio - 1|','Delta MAE (kN)','Delta RMSE (kN)'};
PairedBootstrap = table(string(deltaNames'),mean(deltaBoot,1)', ...
    prctile(deltaBoot,2.5)',prctile(deltaBoot,97.5)',100*mean(deltaBoot<0,1)', ...
    'VariableNames',{'Metric','Delta_MINN_minus_ANN','CI95_Lower','CI95_Upper', ...
    'Percent_resamples_MINN_better'});
writetable(PairedBootstrap,excelFile,'Sheet','PAIRED_BOOTSTRAP');

%% Optional external uncertainty validation (no interval recalibration).
% Select ANN_MINN_5Fold_External_Predictions.xlsx once. The script reads
% the EXTERNAL_PREDICTIONS sheet and the MINN/ANN columns automatically.
ExternalRows = table(); External = struct([]);
 [fExt,pExt] = uigetfile({'*.xlsx','External prediction workbook (*.xlsx)'}, ...
    'Optional: select ANN_MINN_5Fold_External_Predictions.xlsx (Cancel to skip)');
if ~isequal(fExt,0)
    extFile = fullfile(pExt,fExt);
    Text = readtable(extFile,'Sheet','EXTERNAL_PREDICTIONS','VariableNamingRule','preserve');
    required = {'N','MINN_pred_5Fold_Mean','ANN_pred_5Fold_Mean'};
    if ~all(ismember(required,Text.Properties.VariableNames))
        error('Select the workbook exported by ANN_MINN_5Fold_Predict_External.m.');
    end
    yExtAll = Text.N;
    for m = 1:numel(Result)
        predVar = [Result(m).name '_pred_5Fold_Mean'];
        pExtVal = Text.(predVar);
        valid = isfinite(yExtAll) & isfinite(pExtVal) & yExtAll>0;
        yExt = yExtAll(valid); pExtVal = pExtVal(valid);
        loExt = pExtVal-Result(m).q95; hiExt = pExtVal+Result(m).q95;
        widthExt = hiExt-loExt; coveredExt = yExt>=loExt & yExt<=hiExt;
        TExtPI = table(yExt,pExtVal,loExt,hiExt,widthExt,coveredExt, ...
            'VariableNames',{'N_exp_kN','N_pred_kN','PI95_Lower_kN','PI95_Upper_kN', ...
            'PI95_Width_kN','Covered_by_PI95'});
        ExternalRows = [ExternalRows; table(string(Result(m).name),numel(yExt),Result(m).q95, ...
            100*mean(coveredExt),mean(widthExt),median(widthExt), ...
            'VariableNames',{'Model','N_External','Residual_Radius_kN', ...
            'External_Coverage_percent','Mean_PI_Width_kN','Median_PI_Width_kN'})]; %#ok<AGROW>
        External(end+1).name = Result(m).name; %#ok<SAGROW>
        External(end).y = yExt; External(end).p = pExtVal; External(end).lo = loExt;
        External(end).hi = hiExt; External(end).covered = coveredExt; External(end).T = TExtPI;
    end
end
if ~isempty(External)
    writetable(ExternalRows,excelFile,'Sheet','PI95_EXTERNAL_SUMMARY');
    for m = 1:numel(External)
        writetable(External(m).T,excelFile,'Sheet',['PI95_EXTERNAL_' External(m).name]);
    end
end

% Figure 1: clean log-log parity plot for PI95 coverage.
% PI bounds are reported in Excel; error bars are omitted here to avoid visual
% distortion on log axes. PI radii remain calibrated from internal OOF residuals.
f1 = figure('Color','w','Position',[90 100 1350 620]);
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
for m = 1:numel(Result)
    ax = nexttile; R = Result(m); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    datasets = {R}; colors = testColor; markers = {'o'};
    if ~isempty(External)
        datasets{end+1} = External(m); colors(end+1,:) = externalColor; markers{end+1} = 'd';
    end
    allPositive = [];
    for d = 1:numel(datasets)
        S = datasets{d};
        allPositive = [allPositive; S.y(S.y>0); S.p(S.p>0)]; %#ok<AGROW>
    end
    limLow = min(allPositive)*0.75; limHigh = max(allPositive)*1.18;
    hDataset = gobjects(numel(datasets),1); hOut = gobjects(numel(datasets),1);
    for d = 1:numel(datasets)
        S = datasets{d}; c = colors(d,:);
        % Dataset colour/marker remains visible; red cross overlays only the PI95 failures.
        hDataset(d) = scatter(ax,S.y,S.p,31,c,markers{d}, ...
            'filled','MarkerFaceAlpha',0.78);
        hOut(d) = scatter(ax,S.y(~S.covered),S.p(~S.covered),50,[0.85 0.10 0.10],'x', ...
            'LineWidth',1.65,'HandleVisibility','off');
    end
    hRef = plot(ax,[limLow limHigh],[limLow limHigh],'k--','LineWidth',1.25);
    set(ax,'XScale','log','YScale','log','XLim',[limLow limHigh],'YLim',[limLow limHigh]); axis(ax,'square');
    xlabel(ax,'Experimental N_u (kN, log scale)'); ylabel(ax,'Predicted N_u (kN, log scale)');
    info = sprintf('Testing: %d/%d (%.2f%%)', ...
        sum(R.covered),numel(R.y),100*mean(R.covered));
    if ~isempty(External)
        E = External(m);
        info = sprintf('%s\nExternal: %d/%d (%.2f%%)\nMean PI width = %.0f kN', ...
            info,sum(E.covered),numel(E.y),100*mean(E.covered),mean(E.hi-E.lo));
    else
        info = sprintf('%s\nMean PI width = %.0f kN', ...
            info,mean(R.hi-R.lo));
    end
    title(ax,sprintf('(%c) %s',char(96+m),R.name),'FontWeight','bold');
    text(ax,0.04,0.95,sprintf('Residual-calibrated PI95\n%s',info), ...
        'Units','normalized','VerticalAlignment','top','FontWeight','bold', ...
        'BackgroundColor','w','EdgeColor',[.65 .65 .65],'Margin',7);
    if ~isempty(External)
        legend(ax,[hRef hDataset(1) hDataset(2) hOut(1)], ...
            {'1:1 line','Testing','External','Not covered by PI95'},'Location','southeast','Box','on');
    else
        legend(ax,[hRef hDataset(1) hOut(1)],{'1:1 line','Testing','Not covered by PI95'},'Location','southeast','Box','on');
    end
    set(ax,'FontName','Times New Roman','FontSize',14,'LineWidth',0.9);
end
title(tl,'Residual-calibrated PI95 coverage: independent testing and external validation', ...
    'FontName','Times New Roman','FontWeight','bold','FontSize',16);
exportgraphics(f1,fullfile(outDir,'Fig_PI95_Combined_Testing_External_Log.png'),'Resolution',1200);
exportgraphics(f1,fullfile(outDir,'Fig_PI95_Combined_Testing_External_Log.tif'),'Resolution',1200);

% Figure 1b: compact reliability summary for Section 3.3.2.
fPI = figure('Color','w','Position',[110 120 1100 460]);
tlPI = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
modelLabels = cellstr(PIrows.Model);
coverage = PIrows.Empirical_Coverage_percent;
meanWidth = PIrows.Mean_PI_Width_kN;
for panel = 1:2
    ax = nexttile; hold(ax,'on'); box(ax,'on');
    if panel==1
        vals = coverage; yLab = 'Empirical PI95 coverage (%)';
        titleText = '(a) Interval coverage reliability';
        b = bar(ax,1:height(PIrows),vals,0.58,'FaceColor','flat','EdgeColor','none');
        b.CData = modelColor;
        yline(ax,95,'k--','LineWidth',1.4,'Label','Nominal 95%', ...
            'LabelHorizontalAlignment','left','LabelVerticalAlignment','bottom');
        ylim(ax,[90 100]); yticks(ax,90:2:100);
        for k=1:numel(vals)
            text(ax,k,vals(k)+0.35,sprintf('%.2f%%',vals(k)), ...
                'HorizontalAlignment','center','FontWeight','bold','Color',modelColor(k,:));
        end
    else
        vals = meanWidth; yLab = 'Mean PI95 width (kN)';
        titleText = '(b) Prediction-interval sharpness';
        b = bar(ax,1:height(PIrows),vals,0.58,'FaceColor','flat','EdgeColor','none');
        b.CData = modelColor;
        pad = max(vals)*0.20; ylim(ax,[0 max(vals)+pad]);
        for k=1:numel(vals)
            text(ax,k,vals(k)+0.04*max(vals),sprintf('%.0f kN',vals(k)), ...
                'HorizontalAlignment','center','FontWeight','bold','Color',modelColor(k,:));
        end
    end
    xlim(ax,[0.4 2.6]); xticks(ax,1:2); xticklabels(ax,modelLabels);
    ylabel(ax,yLab); title(ax,titleText,'FontWeight','bold');
    grid(ax,'on'); ax.YGrid='on'; ax.XGrid='off';
    set(ax,'FontName','Times New Roman','FontSize',14,'LineWidth',0.9);
end
title(tlPI,'Reliability and sharpness of residual-calibrated 95% prediction intervals', ...
    'FontName','Times New Roman','FontWeight','bold','FontSize',16);
exportgraphics(fPI,fullfile(outDir,'Fig_PI95_Reliability_Summary.png'),'Resolution',1200);
exportgraphics(fPI,fullfile(outDir,'Fig_PI95_Reliability_Summary.tif'),'Resolution',1200);

% Figure 2: Bootstrap 95% CIs of aggregate accuracy metrics.
f2 = figure('Color','w','Position',[90 120 1380 430]);
tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
metricOrder = {'Mean prediction ratio','MAE (kN)','RMSE (kN)'};
for j = 1:3
    nexttile; hold on; box on; grid on;
    Tj = CIrows(CIrows.Metric==metricOrder{j},:);
    for m = 1:height(Tj)
        lower = Tj.Estimate(m)-Tj.CI95_Lower(m);
        upper = Tj.CI95_Upper(m)-Tj.Estimate(m);
        errorbar(m,Tj.Estimate(m),lower,upper,'o','Color',modelColor(m,:), ...
            'MarkerFaceColor',modelColor(m,:),'MarkerSize',8,'LineWidth',1.7,'CapSize',10);
    end
    if j==1, yline(1,'k--','LineWidth',1.1,'HandleVisibility','off'); end
    xlim([0.5 2.5]); xticks(1:2); xticklabels(cellstr(Tj.Model));
    ylabel(strrep(metricOrder{j},' (',' ('),'FontWeight','bold');
    title(sprintf('(%c) %s',char(96+j),metricOrder{j}),'FontWeight','bold');
    set(gca,'FontName','Times New Roman','FontSize',11,'LineWidth',0.9);
end
title(tl,sprintf('Non-parametric bootstrap 95%% confidence intervals (%d resamples)',B), ...
    'FontName','Times New Roman','FontWeight','bold','FontSize',15);
exportgraphics(f2,fullfile(outDir,'Fig_Bootstrap_CI_MINN_ANN.png'),'Resolution',1200);
exportgraphics(f2,fullfile(outDir,'Fig_Bootstrap_CI_MINN_ANN.tif'),'Resolution',1200);

% Figure 3: publication-style bootstrap distributions (violin + CI + estimate).
f3 = figure('Color','w','Position',[90 120 1380 470]);
tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
panelTitles = {'Mean prediction ratio','MAE (kN)','RMSE (kN)'};
for j = 1:3
    ax = nexttile; hold(ax,'on'); box(ax,'on');
    for m = 1:numel(Result)
        samples = Result(m).metricBoot(:,j);
        estimate = Result(m).metricPoint(j);
        ci = prctile(samples,[2.5 97.5]);
        drawBootstrapViolin(ax,samples,m,modelColor(m,:),estimate,ci);
    end
    if j==1, yline(ax,1,'k--','LineWidth',1.1,'HandleVisibility','off'); end
    xlim(ax,[0.35 2.65]); xticks(ax,1:2); xticklabels(ax,models);
    ylabel(ax,panelTitles{j});
    title(ax,sprintf('(%c) %s',char(96+j),panelTitles{j}),'FontWeight','bold');
    grid(ax,'on'); ax.YGrid='on'; ax.XGrid='off';
    set(ax,'FontName','Times New Roman','FontSize',11,'LineWidth',0.9);
end
title(tl,sprintf('Bootstrap distributions, point estimates, and 95%% confidence intervals (%d resamples)',B), ...
    'FontName','Times New Roman','FontWeight','bold','FontSize',15);
exportgraphics(f3,fullfile(outDir,'Fig_Bootstrap_Violin_CI_MINN_ANN.png'),'Resolution',1200);
exportgraphics(f3,fullfile(outDir,'Fig_Bootstrap_Violin_CI_MINN_ANN.tif'),'Resolution',1200);

% Figure 4: paired bootstrap differences. Negative values favour MINN.
f4 = figure('Color','w','Position',[90 120 1380 470]);
tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
for j = 1:3
    ax = nexttile; hold(ax,'on'); box(ax,'on');
    samples = deltaBoot(:,j); estimate = mean(samples); ci = prctile(samples,[2.5 97.5]);
    colour = [0.35 0.20 0.60];
    drawBootstrapViolin(ax,samples,1,colour,estimate,ci);
    yline(ax,0,'k--','LineWidth',1.1,'HandleVisibility','off');
    xlim(ax,[0.55 1.45]); xticks(ax,[]);
    ylabel(ax,deltaNames{j});
    title(ax,sprintf('(%c) %s',char(96+j),deltaNames{j}),'FontWeight','bold');
    verdict = 'MINN better'; if ci(1)>0, verdict = 'ANN better'; elseif ci(1)<=0 && ci(2)>=0, verdict = 'No clear difference'; end
    text(ax,0.04,0.95,sprintf('95%% CI: [%.3g, %.3g]\n%s',ci(1),ci(2),verdict), ...
        'Units','normalized','VerticalAlignment','top','BackgroundColor','w', ...
        'EdgeColor',[.65 .65 .65],'Margin',5,'FontSize',9,'FontWeight','bold');
    grid(ax,'on'); ax.YGrid='on'; ax.XGrid='off';
    set(ax,'FontName','Times New Roman','FontSize',11,'LineWidth',0.9);
end
title(tl,sprintf('Paired bootstrap comparison: MINN minus ANN (%d matched resamples)',B), ...
    'FontName','Times New Roman','FontWeight','bold','FontSize',15);
exportgraphics(f4,fullfile(outDir,'Fig_PairedBootstrap_Difference_MINN_vs_ANN.png'),'Resolution',1200);
exportgraphics(f4,fullfile(outDir,'Fig_PairedBootstrap_Difference_MINN_vs_ANN.tif'),'Resolution',1200);

fprintf('\nOutputs saved to: %s\n',outDir);
disp(PIrows); disp(CIrows); disp(PairedBootstrap);

function file = chooseResultsWorkbook(resultDir,model)
    d = dir(fullfile(resultDir,'**','*5Fold_Results.xlsx'));
    files = fullfile({d.folder},{d.name});
    if isempty(files), error('No *5Fold_Results.xlsx file found in %s.',resultDir); end
    if numel(files)==1, file=files{1}; return; end
    [idx,ok] = listdlg('PromptString',['Select ' model ' results workbook'], ...
        'SelectionMode','single','ListString',files,'ListSize',[700 320]);
    if ~ok, error('No results workbook selected.'); end
    file=files{idx};
end

function drawBootstrapViolin(ax,samples,xPos,color,estimate,ci)
    % Kernel-density violin, 95% CI, and point estimate; avoids version-
    % dependent violinchart properties and works in standard MATLAB releases.
    [density,yGrid] = ksdensity(samples,'NumPoints',180);
    halfWidth = 0.33*density/max(density);
    fill(ax,[xPos+halfWidth fliplr(xPos-halfWidth)], ...
        [yGrid fliplr(yGrid)],color,'FaceAlpha',0.28,'EdgeColor',color, ...
        'LineWidth',1.1,'HandleVisibility','off');
    % Light raw bootstrap points make the sampling distribution visible.
    showN = min(450,numel(samples)); idx = randperm(numel(samples),showN);
    jitter = (rand(showN,1)-0.5)*0.16;
    scatter(ax,xPos+jitter,samples(idx),7,color,'filled', ...
        'MarkerFaceAlpha',0.13,'MarkerEdgeAlpha',0.13,'HandleVisibility','off');
    plot(ax,[xPos xPos],ci,'-','Color',color,'LineWidth',3.4,'HandleVisibility','off');
    plot(ax,xPos,estimate,'o','MarkerSize',8,'MarkerFaceColor',color, ...
        'MarkerEdgeColor','w','LineWidth',1.0,'HandleVisibility','off');
    text(ax,xPos,ci(2),sprintf('  %.3g',estimate),'FontSize',9, ...
        'VerticalAlignment','bottom','Color',color,'FontWeight','bold');
end

function T = readExternalPredictionFile(file)
    [~,~,ext] = fileparts(file);
    if any(strcmpi(ext,{'.xlsx','.xls'}))
        sheets = sheetnames(file);
        [idx,ok] = listdlg('PromptString','Select the sheet containing external predictions', ...
            'SelectionMode','single','ListString',sheets,'ListSize',[380 280]);
        if ~ok, error('No external-prediction sheet selected.'); end
        T = readtable(file,'Sheet',sheets{idx},'VariableNamingRule','preserve');
    else
        T = readtable(file,'VariableNamingRule','preserve');
    end
end

function [y,p] = chooseExternalColumns(T,model)
    isNum = varfun(@isnumeric,T,'OutputFormat','uniform');
    vars = T.Properties.VariableNames(isNum);
    if numel(vars)<2, error('External prediction file needs at least two numeric columns.'); end
    [idx,ok] = listdlg('PromptString',sprintf('%s external file: select [N_exp, N_pred] in that order',model), ...
        'SelectionMode','multiple','ListString',vars,'ListSize',[480 300]);
    if ~ok || numel(idx)~=2, error('Select exactly two numeric columns: N_exp first, then N_pred.'); end
    y = T.(vars{idx(1)}); p = T.(vars{idx(2)});
end
