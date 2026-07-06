clc;
clear;
close all;

%% ============================================================
% COMBINED SCRIPT
% 1) Normalized SHAP Heatmap
% 2) Mean Elasticity Heatmap
% 3) Sensitivity Diverging Bar Charts
%% ============================================================

%% ============================================================
% USER SETTINGS
%% ============================================================

% If your Excel file name is different, edit this line only.
file = 'Model accuracy.xlsx';

% Backup file name used in the original scripts
if ~isfile(file)
    file = 'Model accuracy.xlsx';
end

% Output main folder
mainOutFolder = 'All_ML_SHAP_Sensitivity_Elasticity_Compare_result';

% Subfolders
outFolderSHAP = fullfile(mainOutFolder,'All_ML_SHAP_Heatmap_Figures');
outFolderELA  = fullfile(mainOutFolder,'All_ML_Heatmap_Figures');
outFolderSEN  = fullfile(mainOutFolder,'Sensitivity_Bar_Charts');

if ~exist(mainOutFolder,'dir'); mkdir(mainOutFolder); end
if ~exist(outFolderSHAP,'dir'); mkdir(outFolderSHAP); end
if ~exist(outFolderELA,'dir'); mkdir(outFolderELA); end
if ~exist(outFolderSEN,'dir'); mkdir(outFolderSEN); end

%% ============================================================
% COMMON ORDER AND LABELS
%% ============================================================

modelList = ["DT","RF","ET",...
             "LightGBM","GBM","CatBoost","AdaBoost","XGBoost",...
             "OLS","Ridge","Lasso",...
             "KNN","ANN","MINN"];

varOrder = ["D","t","Fc","Fy","L_D"];

varLabels = {'D','t','f_c','f_y','L/D'};

fontName = 'Times New Roman';

%% ============================================================
%% PART 1: NORMALIZED SHAP HEATMAP
%% ============================================================

sheetName = 'SHAP';

%% ============================================================
% READ TABLE
%% ============================================================

T = readtable(file,...
    'Sheet',sheetName,...
    'VariableNamingRule','preserve');

%% ============================================================
% COLUMN NAMES
%% ============================================================

colNames = string(T.Properties.VariableNames);

modelCol = colNames(contains(colNames,"Model","IgnoreCase",true));
dataCol  = colNames(contains(colNames,"Data","IgnoreCase",true));
varCol   = colNames(contains(colNames,"Variable","IgnoreCase",true));

normCol = colNames( ...
    contains(colNames,"Normalized","IgnoreCase",true) & ...
    contains(colNames,"SHAP","IgnoreCase",true));

modelCol = modelCol(1);
dataCol  = dataCol(1);
varCol   = varCol(1);
normCol  = normCol(1);

%% ============================================================
% FILTER TEST DATA
%% ============================================================

T = T(strcmpi(string(T.(dataCol)),'Test'), :);

%% ============================================================
% CLEAN TEXT
%% ============================================================

T.(modelCol) = strtrim(string(T.(modelCol)));
T.(varCol)   = strtrim(string(T.(varCol)));

%% ============================================================
% RENAME VARIABLES
%% ============================================================

T.(varCol)(strcmpi(T.(varCol),"f_c")) = "Fc";
T.(varCol)(strcmpi(T.(varCol),"f_y")) = "Fy";
T.(varCol)(strcmpi(T.(varCol),"L/D")) = "L_D";

%% ============================================================
% CONVERT NORMALIZED SHAP
%% ============================================================

if isnumeric(T.(normCol))

    shapValue = T.(normCol);

else

    shapValue = erase(string(T.(normCol)),"%");
    shapValue = str2double(shapValue);

end

% If Excel reads 61.42% as 0.6142, convert to 61.42
if max(shapValue,[],'omitnan') <= 1.5
    shapValue = shapValue * 100;
end

T.NormalizedSHAP_Value = shapValue;

%% ============================================================
% CREATE MATRIX
%% ============================================================

SHAP = nan(length(varOrder), length(modelList));

for i = 1:length(varOrder)

    for j = 1:length(modelList)

        idx = strcmpi(string(T.(modelCol)), modelList(j)) & ...
              strcmpi(string(T.(varCol)), varOrder(i));

        if any(idx)

            SHAP(i,j) = ...
                T.NormalizedSHAP_Value(find(idx,1,'first'));

        end

    end

end

%% ============================================================
% FIGURE
%% ============================================================

figure('Color','w',...
       'Position',[100 80 1600 750]);

imagesc(SHAP);

shading interp

colormap(turbo(512));

ax = gca;

set(ax,...
    'FontName','Times New Roman',...
    'FontSize',15,...
    'LineWidth',1.2,...
    'Box','on');

xticks(1:length(modelList));
xticklabels(modelList);
xtickangle(45);

yticks(1:length(varOrder));
yticklabels(varLabels);

title('Heatmap Comparison of Normalized SHAP Feature Importance among ML Models',...
    'FontName','Times New Roman',...
    'FontSize',22,...
    'FontWeight','bold');

xlabel('ML Models',...
    'FontName','Times New Roman',...
    'FontSize',20,...
    'FontWeight','bold');

ylabel('Input Variables',...
    'FontName','Times New Roman',...
    'FontSize',20,...
    'FontWeight','bold');

colormap(turbo(512));

cb = colorbar;
cb.Label.String = 'Normalized SHAP (%)';
cb.Label.FontName = 'Times New Roman';
cb.Label.FontSize = 20;
cb.Label.FontWeight = 'bold';

caxis([0 100]);

%% ============================================================
% ADD VALUE TEXT
%% ============================================================

for i = 1:length(varOrder)

    for j = 1:length(modelList)

        value = SHAP(i,j);

        if ~isnan(value)

            labelText = sprintf('%.1f%%', value);

            text(j,i,labelText,...
                'HorizontalAlignment','center',...
                'VerticalAlignment','middle',...
                'FontName','Times New Roman',...
                'FontSize',18,...
                'FontWeight','bold',...
                'Color','k');

        end

    end

end

%% ============================================================
% EXPORT
%% ============================================================

exportgraphics(gcf,...
    fullfile(outFolderSHAP,'Normalized_SHAP_All_ML_Heatmap.png'),...
    'Resolution',1200);

exportgraphics(gcf,...
    fullfile(outFolderSHAP,'Normalized_SHAP_All_ML_Heatmap.tif'),...
    'Resolution',1200);

disp('Finished exporting SHAP heatmap figure.')

%% ============================================================
%% PART 2: ELASTICITY HEATMAP
%% ============================================================

sheetName = 'Sensitivity';

%% ============================================================
% READ TABLE
%% ============================================================

T = readtable(file,...
    'Sheet',sheetName,...
    'VariableNamingRule','preserve');

%% ============================================================
% COLUMN NAMES
%% ============================================================

colNames = string(T.Properties.VariableNames);

modelCol = colNames(contains(colNames,"Model","IgnoreCase",true));
dataCol  = colNames(contains(colNames,"Data","IgnoreCase",true));
varCol   = colNames(contains(colNames,"Variable","IgnoreCase",true));

elaCol = colNames(contains(colNames,"Mean_Elasticity","IgnoreCase",true));

modelCol = modelCol(1);
dataCol  = dataCol(1);
varCol   = varCol(1);
elaCol   = elaCol(1);

%% ============================================================
% FILTER TEST DATA
%% ============================================================

T = T(strcmpi(string(T.(dataCol)),'Test'), :);

T.(modelCol) = strtrim(string(T.(modelCol)));
T.(varCol)   = strtrim(string(T.(varCol)));

%% ============================================================
% RENAME VARIABLES
%% ============================================================

T.(varCol)(strcmpi(T.(varCol),"f_c")) = "Fc";
T.(varCol)(strcmpi(T.(varCol),"f_y")) = "Fy";
T.(varCol)(strcmpi(T.(varCol),"L/D")) = "L_D";

%% ============================================================
% CREATE ELASTICITY HEATMAP ONLY
%% ============================================================

metricName = 'Elasticity'; %#ok<NASGU>
metricCol  = elaCol;

figTitle = ...
'Heatmap Comparison of Mean Elasticity among ML Models';

colorLabel = 'Mean Elasticity';

DATA = nan(length(varOrder), length(modelList));

for i = 1:length(varOrder)

    for j = 1:length(modelList)

        idx = strcmpi(string(T.(modelCol)), modelList(j)) & ...
              strcmpi(string(T.(varCol)), varOrder(i));

        if any(idx)

            DATA(i,j) = ...
                T.(metricCol)(find(idx,1,'first'));

        end

    end

end

%% ============================================================
% FIGURE
%% ============================================================

figure('Color','w',...
       'Position',[100 80 1600 700]);

imagesc(DATA);

ax = gca;

set(ax,...
    'FontName','Times New Roman',...
    'FontSize',15,...
    'LineWidth',1.2,...
    'Box','on');

xticks(1:length(modelList));
xticklabels(modelList);
xtickangle(45);

yticks(1:length(varOrder));
yticklabels(varLabels);

title(figTitle,...
    'FontName','Times New Roman',...
    'FontSize',22,...
    'FontWeight','bold');

xlabel('ML Models',...
    'FontName','Times New Roman',...
    'FontSize',20,...
    'FontWeight','bold');

ylabel('Input Variables',...
    'FontName','Times New Roman',...
    'FontSize',20,...
    'FontWeight','bold');

%% ============================================================
% COLORMAP
%% ============================================================

colormap(parula(1024));

cb = colorbar;

cb.Label.String = colorLabel;
cb.Label.FontName = 'Times New Roman';
cb.Label.FontSize = 18;
cb.Label.FontWeight = 'bold';

%% ============================================================
% COLOR LIMIT
%% ============================================================

maxAbs = max(abs(DATA(:)),[],'omitnan');

if maxAbs > 0
    caxis([-maxAbs maxAbs]);
end

%% ============================================================
% ADD VALUE TEXT
%% ============================================================

for i = 1:length(varOrder)

    for j = 1:length(modelList)

        value = DATA(i,j);

        if ~isnan(value)

            labelText = sprintf('%.2f%%', value);

            text(j,i,labelText,...
                'HorizontalAlignment','center',...
                'VerticalAlignment','middle',...
                'FontName','Times New Roman',...
                'FontSize',18,...
                'FontWeight','bold',...
                'Color','k');

        end

    end

end

%% ============================================================
% EXPORT
%% ============================================================

exportgraphics(gcf,...
    fullfile(outFolderELA,...
    'Elasticity_All_ML_Heatmap.png'),...
    'Resolution',1200);

disp('Finished exporting elasticity heatmap.')

%% ============================================================
%% PART 3: SENSITIVITY DIVERGING BAR CHARTS
%% ============================================================

%% ============================================================
% COLUMN NAMES
%% ============================================================

colNames = string(T.Properties.VariableNames);

modelCol = colNames(contains(colNames,"Model","IgnoreCase",true));
dataCol  = colNames(contains(colNames,"Data","IgnoreCase",true));
varCol   = colNames(contains(colNames,"Variable","IgnoreCase",true));

senCol = colNames(contains(colNames,"Mean_Sensitivity","IgnoreCase",true));

modelCol = modelCol(1);
dataCol  = dataCol(1);
varCol   = varCol(1);
senCol   = senCol(1);

% T is already filtered to Test and variable names already renamed in Part 2.

%% ============================================================
% FULL VARIABLE NAMES
%% ============================================================

fullVarNames = {...
    'Outer Diameter (D)',...
    'Steel Tube Thickness (t)',...
    'Concrete Compressive Strength (f_c'')',...
    'Steel Yield Strength (f_y)',...
    'Slenderness Ratio (L/D)'};

%% ============================================================
% CREATE SENSITIVITY MATRIX
%% ============================================================

SEN = nan(length(varOrder), length(modelList));

for i = 1:length(varOrder)

    for j = 1:length(modelList)

        idx = strcmpi(string(T.(modelCol)), modelList(j)) & ...
              strcmpi(string(T.(varCol)), varOrder(i));

        if any(idx)

            SEN(i,j) = T.(senCol)(find(idx,1,'first'));

        end

    end

end

%% ============================================================
% DIVERGING BAR CHART
%% ============================================================

for i = 1:length(varOrder)

    values = SEN(i,:);

    figure('Color','w',...
           'Position',[100 80 1500 1000]);

    b = barh(values,...
        'LineWidth',1.2);

    hold on;

    %% ========================================================
    % COLOR BY MODEL CATEGORY
    %% ========================================================

    b.FaceColor = 'flat';

    for j = 1:length(modelList)

        model = string(modelList(j));

        %% Tree-based
        if any(strcmpi(model,["DT","RF","ET"]))

            b.CData(j,:) = [0.30 0.60 0.90];

        %% Boosting-based
        elseif any(strcmpi(model,...
            ["LightGBM","GBM","CatBoost","AdaBoost","XGBoost"]))

            b.CData(j,:) = [0.20 0.75 0.45];

        %% Linear-based
        elseif any(strcmpi(model,...
            ["OLS","Ridge","Lasso"]))

            b.CData(j,:) = [0.95 0.65 0.20];

        %% KNN
        elseif strcmpi(model,"KNN")

            b.CData(j,:) = [0.75 0.45 0.95];

        %% ANN
        elseif strcmpi(model,"ANN")

            b.CData(j,:) = [0.60 0.60 0.60];

        %% MINN
        elseif strcmpi(model,"MINN")

            b.CData(j,:) = [1.00 0.00 0.00];

        end

    end

    %% ========================================================
    % ZERO LINE
    %% ========================================================

    xline(0,'k','LineWidth',1.8);

    %% ========================================================
    % AXIS STYLE
    %% ========================================================

    set(gca,...
        'FontName',fontName,...
        'FontSize',18,...
        'LineWidth',1.5,...
        'Box','on');

    yticks(1:length(modelList));
    yticklabels(modelList);

    set(gca,'YDir','reverse');

    xlabel('Mean Sensitivity',...
        'FontName',fontName,...
        'FontSize',24,...
        'FontWeight','bold');

    ylabel('Machine Learning Models',...
        'FontName',fontName,...
        'FontSize',24,...
        'FontWeight','bold');

    %% ========================================================
    % FULL TITLE
    %% ========================================================

    title(['Sensitivity Bar Chart for ',...
        fullVarNames{i}],...
        'FontName',fontName,...
        'FontSize',28,...
        'FontWeight','bold');

    grid off;

    %% ========================================================
    % X LIMIT
    %% ========================================================

    xMin = min(values,[],'omitnan');
    xMax = max(values,[],'omitnan');

    xAbs = max(abs([xMin xMax]));

    if xAbs == 0
        xAbs = 1;
    end

    xlim([-1.30*xAbs 1.30*xAbs]);

    %% ========================================================
    % ADD VALUES
    %% ========================================================

    for j = 1:length(modelList)

        value = values(j);

        if isnan(value)
            continue;
        end

        labelText = sprintf('%.2f', value);

        if value >= 0

            xText = value + 0.03*xAbs;
            hAlign = 'left';

        else

            xText = value - 0.03*xAbs;
            hAlign = 'right';

        end

        text(xText,...
             j,...
             labelText,...
             'HorizontalAlignment',hAlign,...
             'VerticalAlignment','middle',...
             'FontName',fontName,...
             'FontSize',18,...
             'FontWeight','bold',...
             'BackgroundColor','none',...
             'Clipping','off');

    end

    %% ========================================================
    % EXPORT
    %% ========================================================

    safeVar = regexprep(varOrder(i),'[^\w]','_');

    exportgraphics(gcf,...
        fullfile(outFolderSEN,...
        ['Diverging_Sensitivity_',char(safeVar),'.png']),...
        'Resolution',1200);

end

disp('Finished exporting all diverging bar charts.')
disp('All SHAP, elasticity, and sensitivity figures were exported successfully.')
