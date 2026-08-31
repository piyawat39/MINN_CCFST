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
from sklearn.neighbors import KNeighborsRegressor


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

SHAP_SAMPLE = 100
BACKGROUND_SAMPLE = 100

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

        "MAPE(%)": np.mean(
            np.abs((y_true - y_pred) / y_true)
        ) * 100,

        "A20": np.mean(
            (ratio >= 0.8) & (ratio <= 1.2)
        ),

        "Mean_Ratio": np.mean(ratio),

        "Std_Ratio": np.std(ratio, ddof=1)
    }


# ===================== Five-Fold Ensemble =====================
class FiveFoldEnsemble:
    """Average predictions from the five final fold models."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        fold_predictions = np.column_stack([
            model.predict(X)
            for model in self.models
        ])
        return np.mean(fold_predictions, axis=1)


# ===================== Sensitivity + Elasticity =====================
def calculate_sensitivity_elasticity(model, X_data, data_name):

    X_sample = X_data.copy()

    if (
        (SENSITIVITY_SAMPLE is not None)
        and
        (len(X_sample) > SENSITIVITY_SAMPLE)
    ):

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

        percent_change_output = (
            delta_y / y_base
        ) * 100

        elasticity_value = (
            percent_change_output
            /
            (DELTA_PERCENT * 100)
        )

        rows.append({

            "Model": "KNN",

            "Data": data_name,

            "Variable": var,

            "Input_Change_%": DELTA_PERCENT * 100,

            "Mean_Sensitivity":
                np.nanmean(sensitivity_value),

            "Median_Sensitivity":
                np.nanmedian(sensitivity_value),

            "Std_Sensitivity":
                np.nanstd(
                    sensitivity_value,
                    ddof=1
                ),

            "Min_Sensitivity":
                np.nanmin(sensitivity_value),

            "Max_Sensitivity":
                np.nanmax(sensitivity_value),

            "Mean_Output_Change_%":
                np.mean(percent_change_output),

            "Median_Output_Change_%":
                np.median(percent_change_output),

            "Std_Output_Change_%":
                np.std(
                    percent_change_output,
                    ddof=1
                ),

            "Min_Output_Change_%":
                np.min(percent_change_output),

            "Max_Output_Change_%":
                np.max(percent_change_output),

            "Mean_Elasticity":
                np.mean(elasticity_value),

            "Median_Elasticity":
                np.median(elasticity_value),

            "Std_Elasticity":
                np.std(
                    elasticity_value,
                    ddof=1
                ),

            "Min_Elasticity":
                np.min(elasticity_value),

            "Max_Elasticity":
                np.max(elasticity_value)
        })

    df = pd.DataFrame(rows)

    df["Abs_Mean_Sensitivity"] = np.abs(
        df["Mean_Sensitivity"]
    )

    df["Abs_Mean_Elasticity"] = np.abs(
        df["Mean_Elasticity"]
    )

    df["Sensitivity_Rank"] = (
        df["Abs_Mean_Sensitivity"]
        .rank(
            ascending=False,
            method="dense"
        )
        .astype(int)
    )

    df["Elasticity_Rank"] = (
        df["Abs_Mean_Elasticity"]
        .rank(
            ascending=False,
            method="dense"
        )
        .astype(int)
    )

    df = df.sort_values(
        by="Abs_Mean_Elasticity",
        ascending=False
    )

    return df


# ===================== SHAP Function =====================
def calculate_shap_knn(model, X_data, data_name):

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

    X_background = X_train.copy()

    if (
        (BACKGROUND_SAMPLE is not None)
        and
        (len(X_background) > BACKGROUND_SAMPLE)
    ):

        X_background = X_background.sample(
            n=BACKGROUND_SAMPLE,
            random_state=random_state
        )

    # Explain the complete five-fold ensemble in the original feature space.
    # KernelExplainer supplies NumPy arrays, so restore the feature names before
    # passing data to the pipelines.
    def predict_from_original_space(X_array):
        X_frame = pd.DataFrame(
            X_array,
            columns=feature_names
        )
        return model.predict(X_frame)

    explainer = shap.KernelExplainer(
        predict_from_original_space,
        X_background.to_numpy()
    )

    shap_values = explainer.shap_values(
        X_sample.to_numpy(),
        nsamples=100
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    shap_df = pd.DataFrame(
        shap_values,
        columns=[
            f"SHAP_{v}"
            for v in feature_names
        ]
    )

    shap_df.insert(0, "Model", "KNN")
    shap_df.insert(1, "Data", data_name)

    shap_summary = pd.DataFrame({

        "Model": "KNN",

        "Data": data_name,

        "Variable": feature_names,

        "Mean_SHAP":
            np.mean(shap_values, axis=0),

        "Mean_Abs_SHAP":
            np.mean(
                np.abs(shap_values),
                axis=0
            ),

        "Std_SHAP":
            np.std(
                shap_values,
                axis=0,
                ddof=1
            ),

        "Min_SHAP":
            np.min(shap_values, axis=0),

        "Max_SHAP":
            np.max(shap_values, axis=0)
    })

    shap_summary["SHAP_Rank"] = (
        shap_summary["Mean_Abs_SHAP"]
        .rank(
            ascending=False,
            method="dense"
        )
        .astype(int)
    )

    shap_summary = shap_summary.sort_values(
        by="Mean_Abs_SHAP",
        ascending=False
    )

    return shap_df, shap_summary


# ===================== Optuna Bayesian Optimization =====================

results = []
fold_results = []

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=random_state
)

print("\n===================== KNN Optuna Bayesian Optimization =====================")
print(f"Bayesian trials = {N_BAYES_TRIALS}")
print(f"K-Fold = {N_SPLITS}")


def objective(trial):

    # KNN search space
    params = {
        "n_neighbors": trial.suggest_int(
            "n_neighbors", 3, 30
        ),
        "weights": trial.suggest_categorical(
            "weights", ["uniform", "distance"]
        ),
        "p": trial.suggest_int(
            "p", 1, 2
        )
    }

    fold_train_metrics = []
    fold_val_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X_train),
        start=1
    ):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        # Preprocessing is fitted only on the training portion of each fold.
        fold_model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(**params))
        ])

        fold_model.fit(X_tr, y_tr)

        train_pred_fold = fold_model.predict(X_tr)
        val_pred_fold = fold_model.predict(X_val)

        train_m = calc_metrics(y_tr, train_pred_fold)
        val_m = calc_metrics(y_val, val_pred_fold)

        fold_train_metrics.append(train_m)
        fold_val_metrics.append(val_m)

        fold_row = {
            "Trial": trial.number + 1,
            "Fold": fold,
            **params
        }

        for metric_name, value in train_m.items():
            fold_row[f"Train_{metric_name}"] = value

        for metric_name, value in val_m.items():
            fold_row[f"Val_{metric_name}"] = value

        fold_results.append(fold_row)

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

    result_row = {
        "Trial": trial.number + 1,
        **params
    }

    for metric_name, value in avg_train.items():
        result_row[f"CV_Mean_Train_{metric_name}"] = value

    for metric_name, value in avg_val.items():
        result_row[f"CV_Mean_Val_{metric_name}"] = value

    for metric_name, value in std_val.items():
        result_row[f"CV_SD_Val_{metric_name}"] = value

    results.append(result_row)

    print(
        f"Trial {trial.number + 1:>3}/{N_BAYES_TRIALS} | "
        f"Mean Val MSE = {avg_val['MSE']:.6f} | "
        f"Mean Val RMSE = {avg_val['RMSE']:.4f} "
        f"± {std_val['RMSE']:.4f} | "
        f"Mean Val R2 = {avg_val['R2']:.6f}"
    )

    # Minimize mean validation MSE over the five folds.
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

# ===================== Train Five Final Fold Models =====================
# Keep the five-fold structure after hyperparameter selection.
# - Training prediction: out-of-fold (OOF); every row is predicted by the one
#   fold model that did not use that row for fitting.
# - Testing prediction: mean prediction from all five final fold models.
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
        ("knn", KNeighborsRegressor(**best_params))
    ])

    fold_model.fit(
        X_train.iloc[train_idx],
        y_train[train_idx]
    )

    # OOF prediction for the held-out fold only.
    train_pred[val_idx] = fold_model.predict(
        X_train.iloc[val_idx]
    )
    train_fold_id[val_idx] = fold

    # Independent test prediction from this fold model.
    test_pred_by_fold[:, fold - 1] = fold_model.predict(
        X_test
    )

    final_models.append(fold_model)

if np.isnan(train_pred).any():
    raise RuntimeError("OOF training predictions are incomplete.")

test_pred = np.mean(test_pred_by_fold, axis=1)
external_pred_by_fold = np.column_stack([model.predict(X_external) for model in final_models])
external_pred = np.mean(external_pred_by_fold, axis=1)

# This wrapper is used by sensitivity, elasticity, and SHAP so that those
# analyses describe the same five-model ensemble used for final test prediction.
best_model = FiveFoldEnsemble(final_models)


# ===================== Metrics =====================
train_metrics = calc_metrics(
    y_train,
    train_pred
)

test_metrics = calc_metrics(
    y_test,
    test_pred
)
external_metrics = calc_metrics(y_external, external_pred) if external_has_target else None


# ===================== Print =====================
print("\n========== BEST PARAMETERS ==========")

for k, v in best_params.items():
    print(f"{k} = {v}")

print(f"\nBest Mean 5-Fold Validation MSE = {best_score:.6f}")


# ===================== Export Prediction =====================
train_out = df_train.copy()

train_out["Actual_N"] = y_train

train_out["Predicted_N"] = train_pred

train_out["OOF_Fold"] = train_fold_id

train_out["Error"] = train_pred - y_train

train_out["Abs_Error"] = np.abs(
    train_out["Error"]
)

train_out["Ratio"] = (
    train_pred / y_train
)

test_out = df_test.copy()

test_out["Actual_N"] = y_test

test_out["Predicted_N"] = test_pred

for fold in range(1, N_SPLITS + 1):
    test_out[f"Predicted_N_Fold_{fold}"] = (
        test_pred_by_fold[:, fold - 1]
    )

test_out["Error"] = test_pred - y_test

test_out["Abs_Error"] = np.abs(
    test_out["Error"]
)

test_out["Ratio"] = (
    test_pred / y_test
)

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


# ===================== Metrics Table =====================
summary = pd.DataFrame({

    "Metric":
        list(train_metrics.keys()),

    "Train":
        list(train_metrics.values()),

    "Test":
        list(test_metrics.values())
})
if external_metrics is not None:
    summary["External"] = [external_metrics[m] for m in summary["Metric"]]


# ===================== Best Parameters =====================
best_params_df = pd.DataFrame({

    "Parameter":
        list(best_params.keys()),

    "Value":
        list(best_params.values())
})

best_params_df.loc[
    len(best_params_df)
] = [
    "Best_Mean_5Fold_Validation_MSE",
    best_score
]

best_params_df.loc[
    len(best_params_df)
] = [
    "Bayesian_Trials",
    N_BAYES_TRIALS
]


# ===================== Search Results =====================
search_results_df = pd.DataFrame(results)

search_results_df = search_results_df.sort_values(
    by="CV_Mean_Val_MSE",
    ascending=True
)

fold_results_df = pd.DataFrame(fold_results)


# ===================== Bayesian Search Information =====================
search_range_df = pd.DataFrame([
    ["n_neighbors", "Integer: 1 to 30"],
    ["weights", "Categorical: uniform, distance"],
    ["p", "Integer: 1 to 2"],
    ["Bayesian_Trials", str(N_BAYES_TRIALS)],
    ["Sampler", "Optuna TPESampler (multivariate=True)"]
], columns=[
    "Parameter",
    "Search_Range"
])


# ===================== Sensitivity + Elasticity =====================
sen_train_df = calculate_sensitivity_elasticity(
    best_model,
    X_train,
    "Train"
)

sen_test_df = calculate_sensitivity_elasticity(
    best_model,
    X_test,
    "Test"
)

sen_elasticity_df = pd.concat(
    [sen_train_df, sen_test_df],
    ignore_index=True
)


# ===================== SHAP =====================
shap_train_df, shap_summary_train_df = calculate_shap_knn(
    best_model,
    X_train,
    "Train"
)

shap_test_df, shap_summary_test_df = calculate_shap_knn(
    best_model,
    X_test,
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


# ===================== Export Excel =====================
out_path = "KNN_5Fold_Optuna_BayesianOpt_SHAP_Sensitivity_Elasticity.xlsx"

with pd.ExcelWriter(
    out_path,
    engine="openpyxl"
) as writer:

    train_out.to_excel(
        writer,
        sheet_name="Train_Data",
        index=False
    )

    test_out.to_excel(
        writer,
        sheet_name="Test_Data",
        index=False
    )

    external_out.to_excel(
        writer,
        sheet_name="External_Data",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="Metrics",
        index=False
    )

    best_params_df.to_excel(
        writer,
        sheet_name="Best_Parameters",
        index=False
    )

    search_results_df.to_excel(
        writer,
        sheet_name="Bayes_Trials",
        index=False
    )

    fold_results_df.to_excel(
        writer,
        sheet_name="KNN_5Fold",
        index=False
    )

    search_range_df.to_excel(
        writer,
        sheet_name="Search_Range",
        index=False
    )

    sen_elasticity_df.to_excel(
        writer,
        sheet_name="Sensitivity_Elasticity",
        index=False
    )

    shap_values_df.to_excel(
        writer,
        sheet_name="SHAP_Values",
        index=False
    )

    shap_summary_df.to_excel(
        writer,
        sheet_name="SHAP_Summary",
        index=False
    )

print(f"\n✅ Exported Excel: {out_path}")
