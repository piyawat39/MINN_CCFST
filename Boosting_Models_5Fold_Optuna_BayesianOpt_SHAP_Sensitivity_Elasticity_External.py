# -*- coding: utf-8 -*-
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
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# ===================== Load data =====================
train_file = 'N_S_Training_DATASET_75.xlsx'
test_file  = 'N_S_Testing_DATASET_25.xlsx'
external_file = 'N_S_Testing_DATASET_External.xlsx'

df_train = pd.read_excel(train_file)
df_test  = pd.read_excel(test_file)


def read_external_data(file_path):
    """Read the external observations, ignoring distribution-summary sheets."""
    workbook = pd.ExcelFile(file_path)

    preferred_sheets = [
        name for name in workbook.sheet_names
        if name.strip().upper() in {"TEST_DATA", "EXTERNAL_DATA", "DATA"}
    ]
    sheets_to_check = preferred_sheets + [
        name for name in workbook.sheet_names
        if name not in preferred_sheets
    ]

    required_columns = {"D", "t", "fy", "fc", "L_D"}
    for sheet_name in sheets_to_check:
        candidate = pd.read_excel(file_path, sheet_name=sheet_name)
        if required_columns.issubset(candidate.columns):
            print(
                f"External dataset: {file_path} | "
                f"sheet: {sheet_name} | rows: {len(candidate)}"
            )
            return candidate

    raise ValueError(
        "No external-data sheet contains all required columns: "
        "D, t, fy, fc, L/D"
    )


df_external = read_external_data(external_file)


# ===================== Split =====================
X_train = df_train.drop(['N'], axis=1)
y_train_real = df_train['N'].values

X_test = df_test.drop(['N'], axis=1)
y_test_real = df_test['N'].values

X_test = X_test.reindex(columns=X_train.columns)

external_has_target = "N" in df_external.columns
X_external = df_external.drop(columns=["N"], errors="ignore")
X_external = X_external.reindex(columns=X_train.columns)

missing_external = [
    col for col in X_train.columns
    if col not in df_external.columns
]
if missing_external:
    raise ValueError(
        "External dataset is missing model input columns: "
        + ", ".join(missing_external)
    )

y_external_real = (
    df_external["N"].to_numpy()
    if external_has_target
    else None
)

feature_names = X_train.columns.tolist()


# ===================== Target transform =====================
USE_LOG_TARGET = True

if USE_LOG_TARGET:
    y_train = np.log(y_train_real)
    y_test  = np.log(y_test_real)
else:
    y_train = y_train_real
    y_test  = y_test_real


# ===================== Settings =====================
random_state = 500
N_SPLITS = 5
N_BAYES_TRIALS = 60

SENSITIVITY_SAMPLE = None
ELASTICITY_SAMPLE = None
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
    """Final ensemble containing one fitted model from each fold."""

    def __init__(self, models):
        self.models = models

    def predict_real(self, X):
        # Convert every fold prediction to the real target scale first, then
        # calculate the arithmetic mean used as the final prediction.
        fold_predictions = []

        for model in self.models:
            pred = model.predict(X)
            if USE_LOG_TARGET:
                pred = np.exp(pred)
            fold_predictions.append(pred)

        return np.mean(
            np.column_stack(fold_predictions),
            axis=1
        )


# ===================== Predict Real Scale =====================
def predict_real(model, X):

    if isinstance(model, FiveFoldEnsemble):
        return model.predict_real(X)

    pred = model.predict(X)

    if USE_LOG_TARGET:
        pred = np.exp(pred)

    return pred


# ===================== Sensitivity + Elasticity =====================
def calculate_sensitivity_elasticity(model, X_data, model_name, data_name):

    X_sample = X_data.copy()

    y_base = predict_real(model, X_sample)

    rows = []

    for var in feature_names:

        X_plus = X_sample.copy()

        delta_x = X_plus[var] * DELTA_PERCENT
        delta_x = delta_x.replace(0, np.nan)

        X_plus[var] = X_plus[var] * (1 + DELTA_PERCENT)

        y_plus = predict_real(model, X_plus)

        delta_y = y_plus - y_base

        sensitivity_value = delta_y / delta_x

        percent_change_output = (delta_y / y_base) * 100

        elasticity_value = percent_change_output / (DELTA_PERCENT * 100)

        rows.append({
            "Model": model_name,
            "Data": data_name,
            "Variable": var,

            "Mean_Sensitivity": np.nanmean(sensitivity_value),
            "Mean_Elasticity": np.mean(elasticity_value)
        })

    sen_df = pd.DataFrame(rows)

    return sen_df


# ===================== SHAP Function =====================
def calculate_shap(model, X_data, model_name, data_name):

    X_sample = X_data.copy()

    if (
        (SHAP_SAMPLE is not None)
        and
        (len(X_sample) > SHAP_SAMPLE)
    ):
        X_sample = X_sample.sample(
            n=SHAP_SAMPLE,
            random_state=random_state
        )

    # Calculate SHAP separately for all five fitted fold models and average
    # their SHAP values. This preserves fast TreeExplainer support and makes
    # the reported interpretation represent the complete five-fold ensemble.
    shap_values_by_fold = []

    for fold_model in model.models:
        preprocessor = fold_model[:-1]
        fitted_model = fold_model.named_steps["model"]

        X_transformed = preprocessor.transform(X_sample)
        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        try:
            explainer = shap.TreeExplainer(fitted_model)
            fold_shap_values = explainer.shap_values(X_transformed_df)
        except Exception:
            explainer = shap.Explainer(
                fitted_model.predict,
                X_transformed_df
            )
            fold_shap_values = explainer(X_transformed_df).values

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
        "Mean_Abs_SHAP": np.mean(np.abs(shap_values), axis=0)
    })

    shap_summary["SHAP_Rank"] = shap_summary[
        "Mean_Abs_SHAP"
    ].rank(
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
    """Define Bayesian/TPE search spaces for each boosting model."""

    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.20, log=True
            )
        }

    elif model_name == "GBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.20, log=True
            )
        }

    elif model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 1200),
            "depth": trial.suggest_int("depth", 2, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.20, log=True
            )
        }

    elif model_name == "AdaBoost":
        return {
            "tree_depth": trial.suggest_int("tree_depth", 2, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 1.0, log=True
            )
        }

    elif model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.20, log=True
            )
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ===================== 5-Fold CV Evaluation =====================

def evaluate_params_cv(
    model_name,
    base_model,
    params,
    trial_number,
    kf,
    fold_results_all
):

    fold_train_metrics = []
    fold_val_metrics = []

    X_train_cv = X_train.reset_index(drop=True)
    y_train_real_cv = np.asarray(y_train_real)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(X_train_cv),
        start=1
    ):

        X_tr = X_train_cv.iloc[train_idx].copy()
        X_val = X_train_cv.iloc[val_idx].copy()

        y_tr_real = y_train_real_cv[train_idx]
        y_val_real = y_train_real_cv[val_idx]

        if USE_LOG_TARGET:
            y_tr_fit = np.log(y_tr_real)
        else:
            y_tr_fit = y_tr_real

        # params.copy() is important because AdaBoost removes tree_depth.
        fold_params = params.copy()

        model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", base_model(**fold_params))
        ])

        model.fit(X_tr, y_tr_fit)

        train_pred_fold = predict_real(model, X_tr)
        val_pred_fold = predict_real(model, X_val)

        tr_m = calc_metrics(y_tr_real, train_pred_fold)
        va_m = calc_metrics(y_val_real, val_pred_fold)

        fold_train_metrics.append(tr_m)
        fold_val_metrics.append(va_m)

        fold_row = params.copy()
        fold_row.update({
            "Model": model_name,
            "Trial": trial_number,
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
            "Val_Mean_Ratio": va_m["Mean_Ratio"]
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

    return avg_train, avg_val, std_val


# ===================== Optuna Bayesian Optimization =====================

def optimize_model(model_name, base_model):

    print(
        f"\n\n===================== {model_name} "
        f"Optuna Bayesian Optimization ====================="
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

    def objective(trial):

        params = suggest_params(trial, model_name)

        avg_train, avg_val, std_val = evaluate_params_cv(
            model_name=model_name,
            base_model=base_model,
            params=params,
            trial_number=trial.number + 1,
            kf=kf,
            fold_results_all=fold_results_all
        )

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

        # Objective: minimize mean validation MSE across five folds.
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
        f"\nBest 5-Fold CV Validation MSE = "
        f"{best_score:.6f}"
    )

    # ===================== Five Final Fold Models =====================
    # Training uses out-of-fold (OOF) predictions. Testing uses the arithmetic
    # mean of predictions from all five fitted fold models.
    X_train_final = X_train.reset_index(drop=True)
    y_train_real_final = np.asarray(y_train_real)

    final_models = []
    train_pred = np.full(len(X_train_final), np.nan, dtype=float)
    train_fold_id = np.zeros(len(X_train_final), dtype=int)
    test_pred_by_fold = np.zeros((len(X_test), N_SPLITS), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(X_train_final),
        start=1
    ):
        final_params = best_params.copy()

        fold_model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", base_model(**final_params))
        ])

        y_fold_fit = y_train_real_final[train_idx]
        if USE_LOG_TARGET:
            y_fold_fit = np.log(y_fold_fit)

        fold_model.fit(
            X_train_final.iloc[train_idx],
            y_fold_fit
        )

        train_pred[val_idx] = predict_real(
            fold_model,
            X_train_final.iloc[val_idx]
        )
        train_fold_id[val_idx] = fold_idx

        test_pred_by_fold[:, fold_idx - 1] = predict_real(
            fold_model,
            X_test
        )

        final_models.append(fold_model)

    if np.isnan(train_pred).any():
        raise RuntimeError("OOF training predictions are incomplete.")

    test_pred = np.mean(test_pred_by_fold, axis=1)
    best_model = FiveFoldEnsemble(final_models)

    # The external set is predicted only after the final five fold models have
    # been fitted. Its final prediction is the arithmetic mean of all folds.
    external_pred_by_fold = np.column_stack([
        predict_real(fold_model, X_external)
        for fold_model in final_models
    ])
    external_pred = np.mean(external_pred_by_fold, axis=1)

    train_metrics = calc_metrics(y_train_real, train_pred)
    test_metrics = calc_metrics(y_test_real, test_pred)

    external_metrics = (
        calc_metrics(y_external_real, external_pred)
        if external_has_target
        else None
    )

    train_out = df_train.copy()
    train_out["Actual_N"] = y_train_real
    train_out["Predicted_N"] = train_pred
    train_out["OOF_Fold"] = train_fold_id
    train_out["Error"] = train_pred - y_train_real
    train_out["Abs_Error"] = np.abs(train_out["Error"])
    train_out["Ratio"] = train_pred / y_train_real

    test_out = df_test.copy()
    test_out["Actual_N"] = y_test_real
    test_out["Predicted_N"] = test_pred
    for fold_idx in range(1, N_SPLITS + 1):
        test_out[f"Predicted_N_Fold_{fold_idx}"] = (
            test_pred_by_fold[:, fold_idx - 1]
        )
    test_out["Error"] = test_pred - y_test_real
    test_out["Abs_Error"] = np.abs(test_out["Error"])
    test_out["Ratio"] = test_pred / y_test_real

    external_out = df_external.copy()
    if external_has_target:
        external_out["Actual_N"] = y_external_real
    external_out["Predicted_N"] = external_pred
    for fold_idx in range(1, N_SPLITS + 1):
        external_out[f"Predicted_N_Fold_{fold_idx}"] = (
            external_pred_by_fold[:, fold_idx - 1]
        )
    if external_has_target:
        external_out["Error"] = external_pred - y_external_real
        external_out["Abs_Error"] = np.abs(external_out["Error"])
        external_out["Ratio"] = external_pred / y_external_real

    metrics_df = pd.DataFrame({
        "Metric": list(train_metrics.keys()),
        "Train": list(train_metrics.values()),
        "Test": list(test_metrics.values())
    })

    if external_metrics is not None:
        metrics_df["External"] = [
            external_metrics[metric]
            for metric in metrics_df["Metric"]
        ]

    best_params_df = pd.DataFrame({
        "Parameter": list(best_params.keys()),
        "Value": list(best_params.values())
    })

    best_params_df.loc[len(best_params_df)] = [
        "Best_5Fold_CV_Val_MSE",
        best_score
    ]

    best_params_df.loc[len(best_params_df)] = [
        "Bayesian_Trials",
        N_BAYES_TRIALS
    ]

    search_results_df = pd.DataFrame(trial_results).sort_values(
        by="CV_Val_MSE_Mean",
        ascending=True
    )

    fold_results_df = pd.DataFrame(fold_results_all)

    # ===================== Sensitivity + Elasticity =====================
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

    # ===================== SHAP =====================
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

    "LightGBM": {
        "base_model":
            lambda **params: LGBMRegressor(
                random_state=random_state,
                verbosity=-1,
                n_jobs=-1,
                **params
            )
    },

    "GBM": {
        "base_model":
            lambda **params: GradientBoostingRegressor(
                random_state=random_state,
                **params
            )
    },

    "CatBoost": {
        "base_model":
            lambda **params: CatBoostRegressor(
                random_seed=random_state,
                loss_function="RMSE",
                verbose=0,
                **params
            )
    },

    "AdaBoost": {
        "base_model":
            lambda **params: AdaBoostRegressor(
                estimator=DecisionTreeRegressor(
                    max_depth=params.pop("tree_depth"),
                    random_state=random_state
                ),
                random_state=random_state,
                **params
            )
    },

    "XGBoost": {
        "base_model":
            lambda **params: XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
                **params
            )
    }
}


# ===================== Run All Models =====================
all_outputs = {}

all_metrics = []
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

    temp_metrics = metrics_df.copy()

    temp_metrics.insert(0, "Model", model_name)

    all_metrics.append(temp_metrics)

    all_sen_elasticity.append(sen_elasticity_df)

    all_shap_values.append(shap_values_df)

    all_shap_summary.append(shap_summary_df)


# ===================== Combined Tables =====================
combined_metrics = pd.concat(
    all_metrics,
    ignore_index=True
)

combined_sen_elasticity = pd.concat(
    all_sen_elasticity,
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
out_path = "Boosting_Models_5Fold_Optuna_BayesianOpt_SHAP_Sensitivity_Elasticity.xlsx"

with pd.ExcelWriter(
    out_path,
    engine="openpyxl"
) as writer:

    combined_metrics.to_excel(
        writer,
        sheet_name="All_Metrics",
        index=False
    )

    combined_sen_elasticity.to_excel(
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

    for model_name, output in all_outputs.items():

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
