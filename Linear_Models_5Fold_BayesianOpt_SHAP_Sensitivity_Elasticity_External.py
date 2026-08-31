# -*- coding: utf-8 -*-
"""
Linear Models + Grid Search + Train MSE Optimization
Models: LinearRegression, Ridge, Lasso
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

from sklearn.linear_model import LinearRegression, Ridge, Lasso


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
N_TRIALS = 60
SENSITIVITY_SAMPLE = None
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

        elasticity_value = (
            percent_change_output
            /
            (DELTA_PERCENT * 100)
        )

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

    df = pd.DataFrame(rows)

    df["Abs_Mean_Sensitivity"] = np.abs(df["Mean_Sensitivity"])
    df["Abs_Mean_Elasticity"] = np.abs(df["Mean_Elasticity"])

    df["Sensitivity_Rank"] = (
        df["Abs_Mean_Sensitivity"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    df["Elasticity_Rank"] = (
        df["Abs_Mean_Elasticity"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    total_elasticity = df["Abs_Mean_Elasticity"].sum()

    df["Normalized_Elasticity"] = (
        df["Abs_Mean_Elasticity"] / total_elasticity
    ) * 100

    df["Normalized_Elasticity_%"] = (
        df["Normalized_Elasticity"]
        .map(lambda x: f"{x:.2f}%")
    )

    df = df.sort_values(
        by="Abs_Mean_Elasticity",
        ascending=False
    )

    return df


# ===================== SHAP Function =====================
def calculate_shap_linear(model, X_data, model_name, data_name):

    # Calculate SHAP for every final fold model and average the five SHAP
    # matrices so the interpretation represents the complete ensemble.
    shap_values_by_fold = []

    for fold_model in model.models:
        preprocessor = fold_model[:-1]
        regressor = fold_model.named_steps["regressor"]

        X_transformed = preprocessor.transform(X_data)

        explainer = shap.LinearExplainer(
            regressor,
            X_transformed
        )

        fold_shap_values = np.asarray(
            explainer.shap_values(X_transformed)
        )
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

    shap_summary["SHAP_Rank"] = (
        shap_summary["Mean_Abs_SHAP"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    total_shap = shap_summary["Mean_Abs_SHAP"].sum()

    shap_summary["Normalized_SHAP"] = (
        shap_summary["Mean_Abs_SHAP"] / total_shap
    ) * 100

    shap_summary["Normalized_SHAP_%"] = (
        shap_summary["Normalized_SHAP"]
        .map(lambda x: f"{x:.2f}%")
    )

    shap_summary = shap_summary.sort_values(
        by="Mean_Abs_SHAP",
        ascending=False
    )

    return shap_df, shap_summary


# ===================== Bayesian Search Space =====================
def suggest_params(trial, model_name):
    """
    Bayesian/TPE search spaces.
    Continuous regularization parameters are searched on a log scale.
    """

    if model_name == "LinearRegression":
        return {
            "fit_intercept": trial.suggest_categorical(
                "fit_intercept", [True, False]
            )
        }

    elif model_name == "Ridge":
        return {
            "alpha": trial.suggest_float(
                "alpha", 1e-4, 100.0, log=True
            )
        }

    elif model_name == "Lasso":
        return {
            "alpha": trial.suggest_float(
                "alpha", 1e-4, 100.0, log=True
            ),
            "max_iter": trial.suggest_categorical(
                "max_iter", [2500, 5000, 7500, 10000]
            )
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ===================== Optimization Function: Bayesian + 5-Fold CV =====================
def optimize_model(model_name, base_model):

    print(
        f"\n\n===================== "
        f"{model_name} Bayesian Optimization + 5-Fold CV "
        f"====================="
    )

    print(f"Bayesian trials = {N_TRIALS}")
    print(f"K-Fold = {N_SPLITS}")

    trial_rows = []
    fold_rows = []

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=random_state
    )

    def objective(trial):

        params = suggest_params(trial, model_name)

        fold_train_metrics = []
        fold_val_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            kf.split(X_train), start=1
        ):

            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]

            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]

            # Preprocessing is fitted within each fold only.
            model = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", base_model(**params))
            ])

            model.fit(X_tr, y_tr)

            train_pred = model.predict(X_tr)
            val_pred = model.predict(X_val)

            train_m = calc_metrics(y_tr, train_pred)
            val_m = calc_metrics(y_val, val_pred)

            fold_train_metrics.append(train_m)
            fold_val_metrics.append(val_m)

            row = {
                "Trial": trial.number + 1,
                "Fold": fold,
                **params
            }

            for metric_name, value in train_m.items():
                row[f"Train_{metric_name}"] = value

            for metric_name, value in val_m.items():
                row[f"Val_{metric_name}"] = value

            fold_rows.append(row)

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

        trial_row = {
            "Trial": trial.number + 1,
            **params
        }

        for metric_name, value in avg_train.items():
            trial_row[f"CV_Mean_Train_{metric_name}"] = value

        for metric_name, value in avg_val.items():
            trial_row[f"CV_Mean_Val_{metric_name}"] = value

        for metric_name, value in std_val.items():
            trial_row[f"CV_SD_Val_{metric_name}"] = value

        trial_rows.append(trial_row)

        print(
            f"Trial {trial.number + 1:>3}/{N_TRIALS} | "
            f"Mean Val MSE = {avg_val['MSE']:.6f} | "
            f"Mean Val RMSE = {avg_val['RMSE']:.4f} "
            f"± {std_val['RMSE']:.4f} | "
            f"Mean Val R2 = {avg_val['R2']:.6f}"
        )

        # Objective: minimize the mean validation MSE across 5 folds.
        return avg_val["MSE"]


    # TPE is a sequential Bayesian optimization method.
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
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    best_params = study.best_params.copy()
    best_score = study.best_value

    print(f"\n========== BEST PARAMETERS: {model_name} ==========")

    for k, v in best_params.items():
        print(f"{k} = {v}")

    print(
        f"\nBest Mean 5-Fold Validation MSE = "
        f"{best_score:.6f}"
    )

    # ===================== Five Final Fold Models =====================
    # Training prediction is out-of-fold (OOF). Testing prediction is the
    # arithmetic mean of predictions from the five fitted fold models.
    final_models = []
    train_pred = np.full(len(X_train), np.nan, dtype=float)
    train_fold_id = np.zeros(len(X_train), dtype=int)
    test_pred_by_fold = np.zeros((len(X_test), N_SPLITS), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X_train),
        start=1
    ):
        fold_model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", base_model(**best_params))
        ])

        fold_model.fit(
            X_train.iloc[train_idx],
            y_train[train_idx]
        )

        train_pred[val_idx] = fold_model.predict(
            X_train.iloc[val_idx]
        )
        train_fold_id[val_idx] = fold

        test_pred_by_fold[:, fold - 1] = fold_model.predict(
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

    # ===================== Prediction Tables =====================
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
    for fold in range(1, N_SPLITS + 1):
        test_out[f"Predicted_N_Fold_{fold}"] = (
            test_pred_by_fold[:, fold - 1]
        )
    test_out["Error"] = test_pred - y_test
    test_out["Abs_Error"] = np.abs(test_out["Error"])
    test_out["Ratio"] = test_pred / y_test

    external_out = df_external.copy()
    if external_has_target:
        external_out["Actual_N"] = y_external
    external_out["Predicted_N"] = external_pred
    for fold in range(1, N_SPLITS + 1):
        external_out[f"Predicted_N_Fold_{fold}"] = external_pred_by_fold[:, fold - 1]
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
        "Best_Mean_5Fold_Validation_MSE",
        best_score
    ]

    best_params_df.loc[len(best_params_df)] = [
        "Bayesian_Trials",
        N_TRIALS
    ]

    best_params_df.loc[len(best_params_df)] = [
        "K_Folds",
        N_SPLITS
    ]

    # ===================== Bayesian Trial Results =====================
    search_results_df = pd.DataFrame(trial_rows)

    search_results_df = search_results_df.sort_values(
        by="CV_Mean_Val_MSE",
        ascending=True
    )

    fold_results_df = pd.DataFrame(fold_rows)

    # Save the actual continuous/categorical Bayesian search space.
    if model_name == "LinearRegression":
        search_range_df = pd.DataFrame([
            ("fit_intercept", "[True, False]")
        ], columns=["Parameter", "Search_Range"])

    elif model_name == "Ridge":
        search_range_df = pd.DataFrame([
            ("alpha", "1e-4 to 100, log-uniform")
        ], columns=["Parameter", "Search_Range"])

    elif model_name == "Lasso":
        search_range_df = pd.DataFrame([
            ("alpha", "1e-4 to 100, log-uniform"),
            ("max_iter", "[2500, 5000, 7500, 10000]")
        ], columns=["Parameter", "Search_Range"])

    # ===================== Sensitivity + Elasticity =====================
    sen_elas_train_df = calculate_sensitivity_elasticity(
        best_model,
        X_train,
        model_name,
        "Train"
    )

    sen_elas_test_df = calculate_sensitivity_elasticity(
        best_model,
        X_test,
        model_name,
        "Test"
    )

    sen_elas_df = pd.concat(
        [sen_elas_train_df, sen_elas_test_df],
        ignore_index=True
    )

    # ===================== SHAP =====================
    shap_train_df, shap_summary_train_df = calculate_shap_linear(
        best_model,
        X_train,
        model_name,
        "Train"
    )

    shap_test_df, shap_summary_test_df = calculate_shap_linear(
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
        search_range_df,
        sen_elas_df,
        shap_values_df,
        shap_summary_df
    )


# ===================== Model Configurations =====================
model_configs = {

    "LinearRegression": {
        "base_model":
            lambda **params: LinearRegression(**params)
    },

    "Ridge": {
        "base_model":
            lambda **params: Ridge(
                random_state=random_state,
                **params
            )
    },

    "Lasso": {
        "base_model":
            lambda **params: Lasso(
                random_state=random_state,
                **params
            )
    }
}


# ===================== Run All Models =====================
all_outputs = []

all_metrics = []
all_sen_elas = []

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
        search_range_df,
        sen_elas_df,
        shap_values_df,
        shap_summary_df

    ) = optimize_model(
        model_name=model_name,
        base_model=config["base_model"]
    )

    all_outputs.append({

        "Model": model_name,

        "train_out": train_out,
        "test_out": test_out,
        "external_out": external_out,

        "metrics": metrics_df,
        "best_params": best_params_df,
        "search_results": search_results_df,
        "fold_results": fold_results_df,
        "search_range": search_range_df,

        "sen_elas": sen_elas_df,

        "shap_values": shap_values_df,
        "shap_summary": shap_summary_df
    })

    temp_metrics = metrics_df.copy()

    temp_metrics.insert(0, "Model", model_name)

    all_metrics.append(temp_metrics)

    all_sen_elas.append(sen_elas_df)

    all_shap_values.append(shap_values_df)

    all_shap_summary.append(shap_summary_df)


# ===================== Combined Tables =====================
combined_metrics = pd.concat(
    all_metrics,
    ignore_index=True
)

combined_sen_elas = pd.concat(
    all_sen_elas,
    ignore_index=True
)

combined_shap_values = pd.concat(
    all_shap_values,
    ignore_index=True
)

combined_shap_summary = pd.concat(
    all_shap_summary,
    ignore_index=True
)


# ===================== Export Excel =====================
out_path = "Linear_Models_5Fold_BayesianOpt_SHAP_Sensitivity_Elasticity.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

    combined_metrics.to_excel(
        writer,
        sheet_name="All_Metrics",
        index=False
    )

    combined_sen_elas.to_excel(
        writer,
        sheet_name="All_Sen_Elasticity",
        index=False
    )

    combined_shap_values.to_excel(
        writer,
        sheet_name="All_SHAP_Values",
        index=False
    )

    combined_shap_summary.to_excel(
        writer,
        sheet_name="All_SHAP_Summary",
        index=False
    )

    for output in all_outputs:

        model_name = output["Model"]

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

        output["search_range"].to_excel(
            writer,
            sheet_name=f"{model_name}_Range",
            index=False
        )

        output["sen_elas"].to_excel(
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

print(f"\n✅ Exported Excel: {out_path}")
