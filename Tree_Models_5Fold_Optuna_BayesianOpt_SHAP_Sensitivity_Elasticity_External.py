# -*- coding: utf-8 -*-
"""
Tree-Based Models + Grid Search + 5-Fold Cross-Validation
Models: DecisionTree, RandomForest, ExtraTrees
Selection criterion: minimum mean 5-fold validation MSE
"""

import pandas as pd
import numpy as np
import shap

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import optuna

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


# ===================== Load data =====================
train_file = 'N_S_Training_DATASET_75.xlsx'
test_file  = 'N_S_Testing_DATASET_25.xlsx'
external_file = 'N_S_Testing_DATASET_External.xlsx'

df_train = pd.read_excel(train_file)
df_test  = pd.read_excel(test_file)

def read_external_data(file_path):
    workbook = pd.ExcelFile(file_path)
    preferred = [s for s in workbook.sheet_names if s.strip().upper() in {"TEST_DATA", "EXTERNAL_DATA", "DATA"}]
    for sheet in preferred + [s for s in workbook.sheet_names if s not in preferred]:
        candidate = pd.read_excel(file_path, sheet_name=sheet)
        if {"D", "t", "fy", "fc", "L_D"}.issubset(candidate.columns):
            print(f"External dataset: {file_path} | sheet: {sheet} | rows: {len(candidate)}")
            return candidate
    raise ValueError("No external-data sheet contains D, t, fy, fc, and L/D.")

df_external = read_external_data(external_file)


# ===================== Split =====================
X_train = df_train.drop(['N'], axis=1)
y_train = df_train['N'].values

X_test = df_test.drop(['N'], axis=1)
y_test = df_test['N'].values

X_test = X_test.reindex(columns=X_train.columns)
external_has_target = "N" in df_external.columns
missing_external = [c for c in X_train.columns if c not in df_external.columns]
if missing_external:
    raise ValueError("External dataset is missing: " + ", ".join(missing_external))
X_external = df_external.drop(columns=["N"], errors="ignore").reindex(columns=X_train.columns)
y_external = df_external["N"].to_numpy() if external_has_target else None

feature_names = X_train.columns.tolist()


# ===================== Settings =====================
random_state = 500
N_SPLITS = 5
N_BAYES_TRIALS = 60

SENSITIVITY_SAMPLE = None
SHAP_SAMPLE = None

DELTA_PERCENT = 0.01


# ===================== Metrics Function =====================
def calc_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ratio = y_pred / y_true

    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE(%)": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "A20": np.mean((ratio >= 0.8) & (ratio <= 1.2)),
        "Mean_Ratio": np.mean(ratio),
        "Std_Ratio": np.std(ratio, ddof=1)
    }


# ===================== Five-Fold Ensemble =====================
class FiveFoldEnsemble:
    """Average predictions from the five fitted final fold models."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        fold_predictions = np.column_stack([
            model.predict(X)
            for model in self.models
        ])
        return np.mean(fold_predictions, axis=1)


# ===================== Sensitivity + Elasticity Function =====================
def calculate_sensitivity_elasticity(model, X_data, model_name, data_name):

    X_sample = X_data.copy()

    if (SENSITIVITY_SAMPLE is not None) and (len(X_sample) > SENSITIVITY_SAMPLE):
        X_sample = X_sample.sample(
            n=SENSITIVITY_SAMPLE,
            random_state=random_state
        )

    y_base = model.predict(X_sample)

    rows = []

    for var in feature_names:

        X_plus = X_sample.copy()

        delta_x = X_plus[var] * DELTA_PERCENT
        delta_x = delta_x.replace(0, np.nan)

        X_plus[var] = X_plus[var] * (1 + DELTA_PERCENT)

        y_plus = model.predict(X_plus)

        delta_y = y_plus - y_base

        sensitivity_value = delta_y / delta_x

        percent_change_output = (delta_y / y_base) * 100

        elasticity_value = percent_change_output / (DELTA_PERCENT * 100)

        rows.append({
            "Model": model_name,
            "Data": data_name,
            "Variable": var,
            "Input_Change_%": DELTA_PERCENT * 100,

            "Mean_Sensitivity": np.nanmean(sensitivity_value),
            "Median_Sensitivity": np.nanmedian(sensitivity_value),
            "Std_Sensitivity": np.nanstd(sensitivity_value, ddof=1),
            "Min_Sensitivity": np.nanmin(sensitivity_value),
            "Max_Sensitivity": np.nanmax(sensitivity_value),

            "Mean_Output_Change_%": np.mean(percent_change_output),
            "Median_Output_Change_%": np.median(percent_change_output),
            "Std_Output_Change_%": np.std(percent_change_output, ddof=1),
            "Min_Output_Change_%": np.min(percent_change_output),
            "Max_Output_Change_%": np.max(percent_change_output),

            "Mean_Elasticity": np.mean(elasticity_value),
            "Median_Elasticity": np.median(elasticity_value),
            "Std_Elasticity": np.std(elasticity_value, ddof=1),
            "Min_Elasticity": np.min(elasticity_value),
            "Max_Elasticity": np.max(elasticity_value)
        })

    sen_elasticity_df = pd.DataFrame(rows)

    sen_elasticity_df["Abs_Mean_Sensitivity"] = np.abs(
        sen_elasticity_df["Mean_Sensitivity"]
    )

    sen_elasticity_df["Abs_Mean_Elasticity"] = np.abs(
        sen_elasticity_df["Mean_Elasticity"]
    )

    sen_elasticity_df["Sensitivity_Rank"] = sen_elasticity_df[
        "Abs_Mean_Sensitivity"
    ].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    sen_elasticity_df["Elasticity_Rank"] = sen_elasticity_df[
        "Abs_Mean_Elasticity"
    ].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    sen_elasticity_df = sen_elasticity_df.sort_values(
        by="Abs_Mean_Elasticity",
        ascending=False
    )

    return sen_elasticity_df


# ===================== SHAP Function =====================
def calculate_shap(model, X_data, model_name, data_name):

    X_sample = X_data.copy()

    if (SHAP_SAMPLE is not None) and (len(X_sample) > SHAP_SAMPLE):
        X_sample = X_sample.sample(
            n=SHAP_SAMPLE,
            random_state=random_state
        )

    # Calculate SHAP for each final fold model and average all five matrices.
    # The reported interpretation therefore represents the full ensemble.
    shap_values_by_fold = []

    for fold_model in model.models:
        preprocessor = fold_model[:-1]
        fitted_model = fold_model.named_steps["regressor"]

        X_transformed = preprocessor.transform(X_sample)
        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        explainer = shap.TreeExplainer(fitted_model)
        fold_shap_values = explainer.shap_values(X_transformed_df)

        if isinstance(fold_shap_values, list):
            fold_shap_values = fold_shap_values[0]

        fold_shap_values = np.asarray(fold_shap_values)

        if fold_shap_values.ndim == 3:
            fold_shap_values = fold_shap_values[:, :, 0]

        shap_values_by_fold.append(fold_shap_values)

    shap_values = np.mean(
        np.stack(shap_values_by_fold, axis=0),
        axis=0
    )

    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"SHAP_{v}" for v in feature_names]
    )

    shap_df.insert(0, "Model", model_name)
    shap_df.insert(1, "Data", data_name)

    shap_summary = pd.DataFrame({
        "Model": model_name,
        "Data": data_name,
        "Variable": feature_names,
        "Mean_SHAP": np.mean(shap_values, axis=0),
        "Mean_Abs_SHAP": np.mean(np.abs(shap_values), axis=0),
        "Std_SHAP": np.std(shap_values, axis=0, ddof=1),
        "Min_SHAP": np.min(shap_values, axis=0),
        "Max_SHAP": np.max(shap_values, axis=0)
    })

    shap_summary["SHAP_Rank"] = shap_summary["Mean_Abs_SHAP"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    shap_summary = shap_summary.sort_values(
        by="Mean_Abs_SHAP",
        ascending=False
    )

    return shap_df, shap_summary


# ===================== Optuna Search Space =====================

def suggest_params(trial, model_name):

    if model_name == "DecisionTree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 10
            )
        }

    elif model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 1200
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 20
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 10
            )
        }

    elif model_name == "ExtraTrees":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 1200
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 20
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 10
            )
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ===================== Optimization Function: Optuna + 5-Fold CV =====================

def optimize_model(model_name, base_model):

    print(
        f"\n\n===================== "
        f"{model_name} Optuna Bayesian Optimization + "
        f"{N_SPLITS}-Fold CV ====================="
    )

    print(f"Bayesian trials = {N_BAYES_TRIALS}")
    print(f"Cross-validation = {N_SPLITS}-Fold")

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=random_state
    )

    trial_results = []
    fold_results_all = []

    X_train_cv = X_train.reset_index(drop=True)
    y_train_cv = np.asarray(y_train)

    def objective(trial):

        params = suggest_params(trial, model_name)

        fold_train_metrics = []
        fold_val_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            kf.split(X_train_cv),
            start=1
        ):

            X_tr = X_train_cv.iloc[train_idx].copy()
            X_val = X_train_cv.iloc[val_idx].copy()

            y_tr = y_train_cv[train_idx]
            y_val = y_train_cv[val_idx]

            fold_params = params.copy()

            model = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", base_model(**fold_params))
            ])

            model.fit(X_tr, y_tr)

            train_pred_fold = model.predict(X_tr)
            val_pred_fold = model.predict(X_val)

            tr_m = calc_metrics(y_tr, train_pred_fold)
            va_m = calc_metrics(y_val, val_pred_fold)

            fold_train_metrics.append(tr_m)
            fold_val_metrics.append(va_m)

            fold_row = params.copy()
            fold_row.update({
                "Model": model_name,
                "Trial": trial.number + 1,
                "Fold": fold_idx,
                "Train_R2": tr_m["R2"],
                "Val_R2": va_m["R2"],
                "Train_MAE": tr_m["MAE"],
                "Val_MAE": va_m["MAE"],
                "Train_MSE": tr_m["MSE"],
                "Val_MSE": va_m["MSE"],
                "Train_RMSE": tr_m["RMSE"],
                "Val_RMSE": va_m["RMSE"],
                "Train_MAPE(%)": tr_m["MAPE(%)"],
                "Val_MAPE(%)": va_m["MAPE(%)"],
                "Train_A20": tr_m["A20"],
                "Val_A20": va_m["A20"],
                "Train_Mean_Ratio": tr_m["Mean_Ratio"],
                "Val_Mean_Ratio": va_m["Mean_Ratio"],
                "Train_Std_Ratio": tr_m["Std_Ratio"],
                "Val_Std_Ratio": va_m["Std_Ratio"]
            })

            fold_results_all.append(fold_row)

        avg_train = {
            k: np.mean([m[k] for m in fold_train_metrics])
            for k in fold_train_metrics[0]
        }

        avg_val = {
            k: np.mean([m[k] for m in fold_val_metrics])
            for k in fold_val_metrics[0]
        }

        std_val = {
            k: np.std([m[k] for m in fold_val_metrics], ddof=1)
            for k in fold_val_metrics[0]
        }

        result_row = params.copy()
        result_row.update({
            "Trial": trial.number + 1,
            "CV_Train_R2_Mean": avg_train["R2"],
            "CV_Val_R2_Mean": avg_val["R2"],
            "CV_Val_R2_SD": std_val["R2"],
            "CV_Train_MAE_Mean": avg_train["MAE"],
            "CV_Val_MAE_Mean": avg_val["MAE"],
            "CV_Val_MAE_SD": std_val["MAE"],
            "CV_Train_MSE_Mean": avg_train["MSE"],
            "CV_Val_MSE_Mean": avg_val["MSE"],
            "CV_Val_MSE_SD": std_val["MSE"],
            "CV_Train_RMSE_Mean": avg_train["RMSE"],
            "CV_Val_RMSE_Mean": avg_val["RMSE"],
            "CV_Val_RMSE_SD": std_val["RMSE"],
            "CV_Train_MAPE_Mean(%)": avg_train["MAPE(%)"],
            "CV_Val_MAPE_Mean(%)": avg_val["MAPE(%)"],
            "CV_Train_A20_Mean": avg_train["A20"],
            "CV_Val_A20_Mean": avg_val["A20"],
            "CV_Val_A20_SD": std_val["A20"],
            "CV_Train_Mean_Ratio_Mean": avg_train["Mean_Ratio"],
            "CV_Val_Mean_Ratio_Mean": avg_val["Mean_Ratio"]
        })

        trial_results.append(result_row)

        print(
            f"Trial {trial.number + 1:>3}/{N_BAYES_TRIALS} | "
            f"Mean Val MSE = {avg_val['MSE']:.6f} | "
            f"Mean Val RMSE = {avg_val['RMSE']:.4f} "
            f"± {std_val['RMSE']:.4f} | "
            f"Mean Val R2 = {avg_val['R2']:.6f}"
        )

        return avg_val["MSE"]


    sampler = optuna.samplers.TPESampler(
        seed=random_state,
        multivariate=True
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=N_BAYES_TRIALS,
        show_progress_bar=True
    )

    best_params = study.best_params.copy()
    best_score = study.best_value

    print(f"\n========== BEST PARAMETERS: {model_name} ==========")

    for k, v in best_params.items():
        print(f"{k} = {v}")

    print(
        f"\nBest {N_SPLITS}-Fold CV Validation MSE = "
        f"{best_score:.6f}"
    )

    # ===================== Five Final Fold Models =====================
    # Training prediction is out-of-fold (OOF). Testing prediction is the
    # arithmetic mean of predictions from the five fitted fold models.
    final_models = []
    train_pred = np.full(len(X_train_cv), np.nan, dtype=float)
    train_fold_id = np.zeros(len(X_train_cv), dtype=int)
    test_pred_by_fold = np.zeros((len(X_test), N_SPLITS), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(X_train_cv),
        start=1
    ):
        final_params = best_params.copy()

        fold_model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", base_model(**final_params))
        ])

        fold_model.fit(
            X_train_cv.iloc[train_idx],
            y_train_cv[train_idx]
        )

        train_pred[val_idx] = fold_model.predict(
            X_train_cv.iloc[val_idx]
        )
        train_fold_id[val_idx] = fold_idx

        test_pred_by_fold[:, fold_idx - 1] = fold_model.predict(
            X_test
        )

        final_models.append(fold_model)

    if np.isnan(train_pred).any():
        raise RuntimeError("OOF training predictions are incomplete.")

    test_pred = np.mean(test_pred_by_fold, axis=1)
    external_pred_by_fold = np.column_stack([model.predict(X_external) for model in final_models])
    external_pred = np.mean(external_pred_by_fold, axis=1)
    best_model = FiveFoldEnsemble(final_models)

    train_metrics = calc_metrics(y_train, train_pred)
    test_metrics = calc_metrics(y_test, test_pred)
    external_metrics = calc_metrics(y_external, external_pred) if external_has_target else None

    train_out = df_train.copy()
    train_out["Actual_N"] = y_train
    train_out["Predicted_N"] = train_pred
    train_out["OOF_Fold"] = train_fold_id
    train_out["Error"] = train_pred - y_train
    train_out["Abs_Error"] = np.abs(train_out["Error"])
    train_out["Ratio"] = train_pred / y_train

    test_out = df_test.copy()
    test_out["Actual_N"] = y_test
    test_out["Predicted_N"] = test_pred
    for fold_idx in range(1, N_SPLITS + 1):
        test_out[f"Predicted_N_Fold_{fold_idx}"] = (
            test_pred_by_fold[:, fold_idx - 1]
        )
    test_out["Error"] = test_pred - y_test
    test_out["Abs_Error"] = np.abs(test_out["Error"])
    test_out["Ratio"] = test_pred / y_test

    external_out = df_external.copy()
    if external_has_target:
        external_out["Actual_N"] = y_external
    external_out["Predicted_N"] = external_pred
    for fold_idx in range(1, N_SPLITS + 1):
        external_out[f"Predicted_N_Fold_{fold_idx}"] = external_pred_by_fold[:, fold_idx - 1]
    if external_has_target:
        external_out["Error"] = external_pred - y_external
        external_out["Abs_Error"] = np.abs(external_out["Error"])
        external_out["Ratio"] = external_pred / y_external

    metrics_df = pd.DataFrame({
        "Metric": list(train_metrics.keys()),
        "Train": list(train_metrics.values()),
        "Test": list(test_metrics.values())
    })
    if external_metrics is not None:
        metrics_df["External"] = [external_metrics[m] for m in metrics_df["Metric"]]

    best_params_df = pd.DataFrame({
        "Parameter": list(best_params.keys()),
        "Value": list(best_params.values())
    })

    best_params_df.loc[len(best_params_df)] = [
        f"Best_{N_SPLITS}Fold_CV_Val_MSE",
        best_score
    ]

    best_params_df.loc[len(best_params_df)] = [
        "Bayesian_Trials",
        N_BAYES_TRIALS
    ]

    search_results_df = pd.DataFrame(trial_results)
    search_results_df = search_results_df.sort_values(
        by="CV_Val_MSE_Mean",
        ascending=True
    )

    fold_results_df = pd.DataFrame(fold_results_all)

    # Interpretation from final model
    sen_train_df = calculate_sensitivity_elasticity(
        best_model,
        X_train,
        model_name,
        "Train"
    )

    sen_test_df = calculate_sensitivity_elasticity(
        best_model,
        X_test,
        model_name,
        "Test"
    )

    sen_elasticity_df = pd.concat(
        [sen_train_df, sen_test_df],
        ignore_index=True
    )

    shap_train_df, shap_summary_train_df = calculate_shap(
        best_model,
        X_train,
        model_name,
        "Train"
    )

    shap_test_df, shap_summary_test_df = calculate_shap(
        best_model,
        X_test,
        model_name,
        "Test"
    )

    shap_values_df = pd.concat(
        [shap_train_df, shap_test_df],
        ignore_index=True
    )

    shap_summary_df = pd.concat(
        [shap_summary_train_df, shap_summary_test_df],
        ignore_index=True
    )

    return (
        train_out,
        test_out,
        external_out,
        metrics_df,
        best_params_df,
        search_results_df,
        fold_results_df,
        sen_elasticity_df,
        shap_values_df,
        shap_summary_df
    )


# ===================== Model Configurations =====================

model_configs = {

    "DecisionTree": {
        "base_model":
            lambda **params: DecisionTreeRegressor(
                random_state=random_state,
                **params
            )
    },

    "RandomForest": {
        "base_model":
            lambda **params: RandomForestRegressor(
                random_state=random_state,
                n_jobs=-1,
                **params
            )
    },

    "ExtraTrees": {
        "base_model":
            lambda **params: ExtraTreesRegressor(
                random_state=random_state,
                n_jobs=-1,
                **params
            )
    }
}


# ===================== Run All Models =====================
all_outputs = {}

all_metrics = []
all_best_params = []
all_sen_elasticity = []
all_shap_values = []
all_shap_summary = []

for model_name, config in model_configs.items():

    (
        train_out,
        test_out,
        external_out,
        metrics_df,
        best_params_df,
        search_results_df,
        fold_results_df,
        sen_elasticity_df,
        shap_values_df,
        shap_summary_df

    ) = optimize_model(
        model_name=model_name,
        base_model=config["base_model"]
    )

    temp_metrics = metrics_df.copy()
    temp_metrics.insert(0, "Model", model_name)
    all_metrics.append(temp_metrics)

    temp_best = best_params_df.copy()
    temp_best.insert(0, "Model", model_name)
    all_best_params.append(temp_best)

    all_sen_elasticity.append(sen_elasticity_df)
    all_shap_values.append(shap_values_df)
    all_shap_summary.append(shap_summary_df)

    all_outputs[model_name] = {
        "train_out": train_out,
        "test_out": test_out,
        "external_out": external_out,
        "metrics": metrics_df,
        "best_params": best_params_df,
        "search_results": search_results_df,
        "fold_results": fold_results_df,
        "sen_elasticity": sen_elasticity_df,
        "shap_values": shap_values_df,
        "shap_summary": shap_summary_df
    }


# ===================== Combined Tables =====================
metrics_df_all = pd.concat(
    all_metrics,
    ignore_index=True
)

best_params_all = pd.concat(
    all_best_params,
    ignore_index=True
)

sen_elasticity_all = pd.concat(
    all_sen_elasticity,
    ignore_index=True
)

shap_values_all = pd.concat(
    all_shap_values,
    ignore_index=True
)

shap_summary_all = pd.concat(
    all_shap_summary,
    ignore_index=True
)


# ===================== Export Excel =====================
out_path = "Tree_Models_5Fold_Optuna_BayesianOpt_SHAP_Sensitivity_Elasticity.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

    metrics_df_all.to_excel(
        writer,
        sheet_name="All_Metrics",
        index=False
    )

    best_params_all.to_excel(
        writer,
        sheet_name="Best_Parameters",
        index=False
    )

    sen_elasticity_all.to_excel(
        writer,
        sheet_name="All_Sen_Elasticity",
        index=False
    )

    shap_values_all.to_excel(
        writer,
        sheet_name="All_SHAP_Values",
        index=False
    )

    shap_summary_all.to_excel(
        writer,
        sheet_name="All_SHAP_Summary",
        index=False
    )

    for model_name, output in all_outputs.items():

        output["train_out"].to_excel(
            writer,
            sheet_name=f"{model_name}_Train",
            index=False
        )

        output["test_out"].to_excel(
            writer,
            sheet_name=f"{model_name}_Test",
            index=False
        )

        output["external_out"].to_excel(
            writer,
            sheet_name=f"{model_name}_External",
            index=False
        )

        output["search_results"].to_excel(
            writer,
            sheet_name=f"{model_name}_BayesTrials",
            index=False
        )

        output["fold_results"].to_excel(
            writer,
            sheet_name=f"{model_name}_5Fold",
            index=False
        )

        output["metrics"].to_excel(
            writer,
            sheet_name=f"{model_name}_Metrics",
            index=False
        )

        output["best_params"].to_excel(
            writer,
            sheet_name=f"{model_name}_BestParam",
            index=False
        )

        output["sen_elasticity"].to_excel(
            writer,
            sheet_name=f"{model_name}_SenElas",
            index=False
        )

        output["shap_values"].to_excel(
            writer,
            sheet_name=f"{model_name}_SHAP",
            index=False
        )

        output["shap_summary"].to_excel(
            writer,
            sheet_name=f"{model_name}_SHAP_Sum",
            index=False
        )

print(f"\n✅ Exported Excel: {out_path}")
