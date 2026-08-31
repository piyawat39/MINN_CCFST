# MINN–CCFST: Code Guide

MATLAB scripts for predicting the axial compressive strength of short circular concrete-filled steel tubular (CCFST) columns. The input variables are outer diameter `D`, tube thickness `t`, steel yield strength `fy`, concrete compressive strength `fc`, and length-to-diameter ratio `L/D`; the output is axial strength `N` (kN).

## Recommended workflow

1. Run `random_into_test_train.m` to create the 75/25 training–testing split.
2. Run the Bayesian-optimization scripts to identify ANN and MINN hyperparameters.
3. Run `ANN_N_S_5_Fold.m` and `MINN_N_S_5_Fold.m` to train and export the five-fold models.
4. Run the analysis, uncertainty, external-validation, or GUI scripts as required.

## Scripts

| Script | Purpose |
| --- | --- |
| `random_into_test_train.m` | Randomly divides the experimental database into 75% training and 25% independent testing sets, and exports descriptive statistics. |
| `ANN_N_S_BayesianOptimization_5Fold.m` | Uses Bayesian optimization with five-fold cross-validation to select ANN learning rate, number of neurons, and number of hidden layers. |
| `MINN_N_S_MultiObjective_Bayesian_5Fold.m` | Uses ParEGO multi-objective Bayesian optimization to select MINN hyperparameters by balancing validation RMSE and mechanics consistency. |
| `ANN_N_S_5_Fold.m` | Trains the five ANN fold models, evaluates the independent test set using the five-model mean prediction, and saves trained models/results. |
| `MINN_N_S_5_Fold.m` | Trains the five mechanics-informed neural-network (MINN) fold models and exports their predictions, losses, and model files. |
| `Linear_Models_5Fold_BayesianOpt_SHAP_Sensitivity_Elasticity_External.py` | Tunes and evaluates linear models (OLS, Ridge, and Lasso) with five-fold cross-validation; exports predictions and SHAP/sensitivity/elasticity results for the independent and external datasets. |
| `KNN_5Fold_Optuna_BayesianOpt_SHAP_Sensitivity_Elasticity_External.py` | Tunes and evaluates the K-nearest-neighbours (KNN) model using Optuna and five-fold cross-validation, including external-data prediction and interpretation results. |
| `Tree_Models_5Fold_Optuna_BayesianOpt_SHAP_Sensitivity_Elasticity_External.py` | Tunes and evaluates tree-based models with Optuna and five-fold cross-validation, including external-data prediction, SHAP, sensitivity, and elasticity analyses. |
| `Boosting_Models_5Fold_Optuna_BayesianOpt_SHAP_Sensitivity_Elasticity_External.py` | Tunes and evaluates boosting models with Optuna and five-fold cross-validation, including external-data prediction and model-interpretation analyses. |
| `Yu_Ding_liang_cal.m` | Calculates axial strengths from the Yu, Ding, and Liang equations and compares them with experiments. |
| `AISC_cal.py` | Calculates CCFST axial capacities using AISC provisions and reports applicability checks and accuracy metrics. |
| `EC4_cal.py` | Calculates CCFST axial capacities using Eurocode 4 (EC4) provisions and reports applicability checks and accuracy metrics. |
| `MINN_ANN_SHAP_SEN_ELAS_Dataset_Cal.m` | Compares ANN and MINN sensitivity, elasticity, and permutation-SHAP results for training and testing data. It uses saved models; no retraining is performed. |
| `MINN_ANN_SHAP_SEN_ELAS_Machanical_based_Cal_ver2.m` | Compares Yu, Liang, ANN, and MINN on the testing data satisfying both equation validity ranges; calculates sensitivity, elasticity, permutation SHAP, and permutation feature importance (PFI). |
| `All_ML_SHAP_Sensitivity_Elasticity_conpare.m` | Creates comparative heatmaps and sensitivity plots for all ML models from exported Excel results. |
| `ANN_MINN_5Fold_Predict_External.m` | Predicts an external Excel dataset using the saved ANN and MINN five-fold ensembles, then exports predictions, accuracy metrics, and a parity plot. |
| `PI95_ResidualCalibrated_MINN_ANN.m` | Produces residual-calibrated 95% prediction intervals (PI95) and bootstrap 95% confidence intervals for completed ANN and MINN results; no retraining is required. |
| `R2_plot_ver2.m` | Creates experimental-versus-predicted plots for the models listed in a model-accuracy Excel workbook. |
| `GUI_CCFST_MINN_ver2.m` | Opens the graphical interface for single-specimen MINN prediction using the mean of the five saved MINN models. |

## Notes

- Place the dataset and the required saved model folders in the locations specified at the beginning of each script.
- Scripts with version labels such as `ver_test` or earlier `Machanical_based_Cal` files are development variants. Use the `ver2` mechanics-based script for the final SHAP/PFI analysis.
- Rename files after download to remove suffixes such as `(1)` or `(9)`, so that they match the names above.
- Python scripts require the packages listed in their import sections (commonly `numpy`, `pandas`, `scikit-learn`, `optuna`, `shap`, and `openpyxl`).
