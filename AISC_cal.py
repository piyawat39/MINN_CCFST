# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================
# INPUT FILES
# ============================================================
train_file = "N_S_Training_DATASET_75.xlsx"
test_file  = "N_S_Testing_DATASET_25.xlsx"

outfile = "AISC_cal.xlsx"

Es = 200000.0  # MPa


# ============================================================
# AISC LIMITATION CHECK
# ============================================================
def limitation_aisc(D, t, Fy, fc, L):

    if min(D, t, Fy, fc, L) <= 0:
        return 0

    if D <= 2 * t:
        return 0

    Ag = math.pi * D**2 / 4.0
    Ac = math.pi * (D - 2*t)**2 / 4.0
    As = Ag - Ac

    cond_fc = 21 <= fc <= 69
    cond_Fy = Fy <= 525
    cond_ratio = As / Ag > 0.01
    cond_dt = D / t < 0.31 * Es / Fy

    return int(cond_fc and cond_Fy and cond_ratio and cond_dt)


# ============================================================
# AISC AXIAL CAPACITY
# ============================================================
def AISC(D, t, Fy, fc, L):

    if limitation_aisc(D, t, Fy, fc, L) == 0:
        return np.nan

    D_t = D / t

    lambda_p = 0.15 * Es / Fy
    lambda_r = 0.19 * Es / Fy
    lambda_max = 0.31 * Es / Fy

    Ag = math.pi * D**2 / 4.0
    Ac = math.pi * (D - 2*t)**2 / 4.0
    As = Ag - Ac

    Ic = math.pi * (D - 2*t)**4 / 64.0
    Is = math.pi * (D**4 - (D - 2*t)**4) / 64.0

    Ec = 0.043 * (2450.0**1.5) * math.sqrt(fc)

    C3 = min(0.9, 0.45 + 3.0 * As / Ag)

    EIaisc = Es * Is + C3 * Ec * Ic

    Pcr = math.pi**2 * EIaisc / L**2

    if Pcr <= 0:
        return np.nan

    Pp = Fy * As + 0.95 * fc * Ac
    Py = Fy * As + 0.70 * fc * Ac

    Fn = 0.72 * Fy / ((D_t * Fy / Es) ** 0.2)

    Pno_slender = Fn * As + 0.70 * fc * Ac
    Pno_compact = Pp
    Pno_noncompact = Pp - ((Pp - Py) * ((D_t - lambda_p)**2) / ((lambda_r - lambda_p)**2))

    if D_t > lambda_max:
        Pno = 0.0
    elif D_t > lambda_r:
        Pno = Pno_slender
    elif D_t > lambda_p:
        Pno = Pno_noncompact
    else:
        Pno = Pno_compact

    ratio = Pno / Pcr

    if ratio > 2.25:
        Pn = 0.877 * Pcr
    else:
        Pn = Pno * (0.658 ** ratio)

    return max(Pn / 1000.0, 0.0)  # kN


# ============================================================
# PROCESS EACH FILE
# ============================================================
def process_file(file_path, dataset_name):

    df = pd.read_excel(file_path)
    df.columns = df.columns.astype(str).str.strip()

    D_col = "D"
    t_col = "t"
    Fy_col = "Fy"
    Fc_col = "Fc"
    target_col = "N"

    required_cols = [D_col, t_col, Fy_col, Fc_col, target_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{dataset_name}: Missing required column: {col}")

    # Create L if not available
    if "L" in df.columns:
        pass
    elif "L_D" in df.columns:
        df["L"] = df["D"] * df["L_D"]
    elif "LD" in df.columns:
        df["L"] = df["D"] * df["LD"]
    elif "L/D" in df.columns:
        df["L"] = df["D"] * df["L/D"]
    else:
        raise ValueError(f"{dataset_name}: Missing column L or L/D")

    df["AISC"] = df.apply(
        lambda row: AISC(
            float(row[D_col]),
            float(row[t_col]),
            float(row[Fy_col]),
            float(row[Fc_col]),
            float(row["L"])
        ),
        axis=1
    )

    df["pass_AISC"] = df["AISC"].notna().astype(int)
    df["AISC_to_Exp"] = df["AISC"] / df[target_col]

    return df


# ============================================================
# METRICS FUNCTION
# ============================================================
def calc_metrics_full(y_true, y_pred):

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (y_true != 0)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ratio = y_pred / y_true

    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "A20": np.mean((ratio >= 0.8) & (ratio <= 1.2)),
        "Mean_Ratio": np.mean(ratio),
        "Std_Ratio": np.std(ratio)
    }


# ============================================================
# PRINT PASS / FAIL
# ============================================================
def print_summary(name, df):

    total = len(df)
    passed = int(df["pass_AISC"].sum())
    failed = total - passed

    print(f"\n===== {name} =====")
    print(f"Total   : {total}")
    print(f"Passed  : {passed} ({passed/total*100:.2f}%)")
    print(f"Failed  : {failed} ({failed/total*100:.2f}%)")


# ============================================================
# RUN
# ============================================================
df_train = process_file(train_file, "TRAIN")
df_test = process_file(test_file, "TEST")

print_summary("AISC TRAIN", df_train)
print_summary("AISC TEST", df_test)

# ============================================================
# METRICS TABLE FORMAT
# ============================================================
metrics_train = calc_metrics_full(df_train["N"], df_train["AISC"])
metrics_test = calc_metrics_full(df_test["N"], df_test["AISC"])

metrics_table = pd.DataFrame([
    ["Train"] + list(metrics_train.values()),
    ["Test"] + list(metrics_test.values())
], columns=[
    "Metric",
    "R2",
    "MAE",
    "MSE",
    "RMSE",
    "A20",
    "Mean_Ratio",
    "Std_Ratio"
])

print("\n===== AISC METRICS TABLE =====")
print(metrics_table.round(6))


# ============================================================
# SUMMARY TABLE
# ============================================================
summary_table = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "Total": [
        len(df_train),
        len(df_test)
    ],
    "Passed_AISC": [
        int(df_train["pass_AISC"].sum()),
        int(df_test["pass_AISC"].sum())
    ],
    "Failed_AISC": [
        int((1 - df_train["pass_AISC"]).sum()),
        int((1 - df_test["pass_AISC"]).sum())
    ],
    "Passed_percent": [
        df_train["pass_AISC"].mean() * 100,
        df_test["pass_AISC"].mean() * 100
    ],
    "Failed_percent": [
        (1 - df_train["pass_AISC"].mean()) * 100,
        (1 - df_test["pass_AISC"].mean()) * 100
    ]
})


# ============================================================
# EXPORT EXCEL
# ============================================================
with pd.ExcelWriter(outfile, engine="openpyxl") as writer:

    df_train.to_excel(writer, index=False, sheet_name="TRAIN")
    df_test.to_excel(writer, index=False, sheet_name="TEST")
    metrics_table.to_excel(writer, index=False, sheet_name="METRICS")
    summary_table.to_excel(writer, index=False, sheet_name="SUMMARY")

    # Auto column width
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]

        for col in ws.columns:
            max_length = 0
            col_letter = col[0].column_letter

            for cell in col:
                try:
                    if cell.value is not None:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass

            ws.column_dimensions[col_letter].width = max_length + 2


print("\nDone!")
print("Exported file:", outfile)