# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:10:09 2026

@author: USER
"""

# ============================================================
# ML model performance + SHAP explainable AI
# Objective salivary hypofunction manuscript
# Spyder-ready integrated code
# ============================================================

# ------------------------------------------------------------
# 0. Package check / installation
# ------------------------------------------------------------

import sys
import subprocess
import importlib.util

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
    "shap": "shap"
}

for import_name, package_name in REQUIRED_PACKAGES.items():
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing missing package: {package_name}")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package_name
        ])

# ------------------------------------------------------------
# 1. Import packages
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve
)

from statsmodels.stats.multitest import multipletests
import shap

print("Python executable:", sys.executable)

# ============================================================
# 2. User settings
# ============================================================

DATA_DIR = Path(
    r"C:\Users\USER\Desktop\2026 연구 VitaminD ESR CRP Prolo\구취\Salivary flow rate 논문"
)

DATA_FILE = DATA_DIR / "Final_analysis_dataset_6class_xerogenic_medication (Final)(2).xlsx"

if not DATA_FILE.exists():
    candidates = list(DATA_DIR.glob("*6class*xerogenic*.xlsx")) + list(DATA_DIR.glob("*.xlsx"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No Excel file found in: {DATA_DIR}")
    DATA_FILE = candidates[0]
    print("Specified file was not found. Using:", DATA_FILE)

OUT_DIR = DATA_DIR / "ML_SHAP_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

# 빠른 테스트는 500, 최종 논문용은 2000 권장
N_BOOTSTRAP = 2000

# 현재 원고의 age-group framework에 맞추려면 True.
# continuous age로 돌리고 싶으면 False.
USE_AGE_GROUP_FOR_ML = True

print("Data file:", DATA_FILE)
print("Output folder:", OUT_DIR)

# ============================================================
# 3. Helper functions
# ============================================================

def clean_colname(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_col(df, candidates, required=True):
    col_map = {clean_colname(c): c for c in df.columns}

    for cand in candidates:
        key = clean_colname(cand)
        if key in col_map:
            return col_map[key]

    for cand in candidates:
        key = clean_colname(cand)
        for c in df.columns:
            if key in clean_colname(c):
                return c

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return None


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def to_binary(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip().str.lower()

    yes_values = {
        "1", "yes", "y", "true", "present",
        "female", "f", "있음", "예"
    }

    no_values = {
        "0", "no", "n", "false", "absent",
        "male", "m", "없음", "아니오"
    }

    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.isin(yes_values)] = 1
    out[s.isin(no_values)] = 0

    return out


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def format_ci(mean_value, low, high, digits=4):
    return f"{mean_value:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    aucs = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)

        if len(np.unique(y_true[idx])) < 2:
            continue

        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))

    if len(aucs) == 0:
        return np.nan, np.nan

    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# ============================================================
# 4. DeLong test functions
# ============================================================

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)

    T = np.zeros(N, dtype=float)
    i = 0

    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1

        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    T2 = np.empty(N, dtype=float)
    T2[J] = T

    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n

    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    delongcov = sx / m + sy / n

    return aucs, delongcov


def delong_roc_test(y_true, pred_one, pred_two):
    y_true = np.asarray(y_true).astype(int)
    pred_one = np.asarray(pred_one)
    pred_two = np.asarray(pred_two)

    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))

    predictions_sorted_transposed = np.vstack((pred_one, pred_two))[:, order]

    aucs, covariance = fast_delong(
        predictions_sorted_transposed,
        label_1_count
    )

    diff = aucs[0] - aucs[1]
    var = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]

    if var <= 0:
        return float(aucs[0]), float(aucs[1]), np.nan

    z = abs(diff) / np.sqrt(var)

    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(z))

    return float(aucs[0]), float(aucs[1]), float(p)


# ============================================================
# 5. Load and prepare data
# ============================================================

df_raw = pd.read_excel(DATA_FILE)

print("Raw data shape:", df_raw.shape)
print("Columns:")
print(df_raw.columns.tolist())

df = df_raw.copy()

COL = {
    "age": find_col(df, ["Age", "AGE"]),
    "female": find_col(df, ["Female sex", "Female", "Sex female", "SEX_FEMA"]),

    "sys_count": find_col(
        df,
        [
            "Number of systemic disease categories",
            "Systemic disease category count",
            "Systemic disease count"
        ]
    ),

    "med_count": find_col(
        df,
        [
            "Number of current medication classes",
            "Current medication class count",
            "Medication class count"
        ]
    ),

    "xer_count": find_col(
        df,
        [
            "High-confidence xerogenic class count",
            "High-confidence xerogenic medication class count",
            "6-class high-confidence xerogenic medication class count"
        ]
    ),

    "ufr": find_col(
        df,
        [
            "UFR (mL/min)",
            "UFR",
            "Unstimulated salivary flow rate"
        ]
    ),

    "sfr": find_col(
        df,
        [
            "SFR (mL/min)",
            "SFR",
            "Stimulated salivary flow rate"
        ]
    ),

    "ufr_hypo": find_col(
        df,
        [
            "UFR-defined hyposalivation",
            "Xerostomia_UFR",
            "UFR hypo"
        ],
        required=False
    ),

    "sfr_hypo": find_col(
        df,
        [
            "SFR-defined hyposalivation",
            "Xerostomia_SFR",
            "SFR hypo"
        ],
        required=False
    ),

    "ph": find_col(
        df,
        [
            "Salivary pH",
            "Salivary pH ",
            "Salivary pH raw",
            "pH"
        ]
    ),

    "buffer": find_col(
        df,
        [
            "Salivary buffer capacity",
            "Buffer capacity"
        ]
    ),

    "vas": find_col(df, ["Visual analog scale", "VAS"]),
    "sticky": find_col(df, ["Sticky saliva binary", "Sticky saliva"]),
    "calculus": find_col(df, ["Dental calculus ordinal", "Dental calculus"]),
    "tongue": find_col(df, ["Tongue coating ordinal", "Tongue coating"]),
    "halitosis": find_col(df, ["Halitosis binary", "Halitosis"]),
}

analysis = pd.DataFrame(index=df.index)

analysis["Age"] = to_numeric(df[COL["age"]])

analysis["Age_group"] = pd.cut(
    analysis["Age"],
    bins=[-np.inf, 39, 59, 74, np.inf],
    labels=[1, 2, 3, 4]
).astype(float)

analysis["Female sex"] = to_binary(df[COL["female"]])

analysis["Number of systemic disease categories"] = to_numeric(
    df[COL["sys_count"]]
)

analysis["Number of current medication classes"] = to_numeric(
    df[COL["med_count"]]
)

analysis["High-confidence xerogenic class count"] = to_numeric(
    df[COL["xer_count"]]
)

analysis["UFR (mL/min)"] = to_numeric(df[COL["ufr"]])
analysis["SFR (mL/min)"] = to_numeric(df[COL["sfr"]])

if COL["ufr_hypo"] is not None:
    analysis["UFR-defined hyposalivation"] = to_binary(df[COL["ufr_hypo"]])
else:
    analysis["UFR-defined hyposalivation"] = (
        analysis["UFR (mL/min)"] < 0.1
    ).astype(float)
    analysis.loc[
        analysis["UFR (mL/min)"].isna(),
        "UFR-defined hyposalivation"
    ] = np.nan

if COL["sfr_hypo"] is not None:
    analysis["SFR-defined hyposalivation"] = to_binary(df[COL["sfr_hypo"]])
else:
    analysis["SFR-defined hyposalivation"] = (
        analysis["SFR (mL/min)"] < 0.7
    ).astype(float)
    analysis.loc[
        analysis["SFR (mL/min)"].isna(),
        "SFR-defined hyposalivation"
    ] = np.nan

analysis["Salivary pH"] = to_numeric(df[COL["ph"]])

analysis.loc[
    (analysis["Salivary pH"] < 5.0) |
    (analysis["Salivary pH"] > 9.0),
    "Salivary pH"
] = np.nan

analysis["Salivary buffer capacity"] = to_numeric(df[COL["buffer"]])
analysis["Visual analog scale"] = to_numeric(df[COL["vas"]])
analysis["Sticky saliva"] = to_numeric(df[COL["sticky"]])
analysis["Dental calculus"] = to_numeric(df[COL["calculus"]])
analysis["Tongue coating"] = to_numeric(df[COL["tongue"]])
analysis["Halitosis"] = to_numeric(df[COL["halitosis"]])

age_feature = "Age_group" if USE_AGE_GROUP_FOR_ML else "Age"

MODEL_FEATURES = {
    "Model 1: Demographic model": [
        age_feature,
        "Female sex"
    ],

    "Model 2: Systemic-medication burden model": [
        age_feature,
        "Female sex",
        "Number of systemic disease categories",
        "Number of current medication classes",
        "High-confidence xerogenic class count"
    ],

    "Model 3: Practical oral-systemic model": [
        age_feature,
        "Female sex",
        "Number of systemic disease categories",
        "Number of current medication classes",
        "High-confidence xerogenic class count",
        "Visual analog scale",
        "Sticky saliva",
        "Dental calculus",
        "Tongue coating",
        "Halitosis"
    ],

    "Model 4: Full salivary oral-systemic model": [
        age_feature,
        "Female sex",
        "Number of systemic disease categories",
        "Number of current medication classes",
        "High-confidence xerogenic class count",
        "Visual analog scale",
        "Sticky saliva",
        "Dental calculus",
        "Tongue coating",
        "Halitosis",
        "Salivary pH",
        "Salivary buffer capacity"
    ],
}

print("\nAnalysis data preview:")
print(analysis.head())

print("\nOutcome counts:")
print("UFR n:", analysis["UFR (mL/min)"].notna().sum())
print("SFR n:", analysis["SFR (mL/min)"].notna().sum())
print("UFR hypo events:", int(np.nansum(analysis["UFR-defined hyposalivation"])))
print("SFR hypo events:", int(np.nansum(analysis["SFR-defined hyposalivation"])))


# ============================================================
# 6. Cross-validated model performance
# ============================================================

def make_regressor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))
    ])


def make_classifier():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])


def evaluate_regression_outcome(outcome_name):
    rows = []
    prediction_frames = []

    for model_name, features in MODEL_FEATURES.items():
        data = analysis[[outcome_name] + features].copy()
        data = data.dropna(subset=[outcome_name])

        y = data[outcome_name].astype(float).values
        X = data[features]

        cv = KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        pred = cross_val_predict(
            make_regressor(),
            X,
            y,
            cv=cv
        )

        rows.append({
            "Outcome": outcome_name,
            "Prediction task": "Regression",
            "Model": model_name,
            "n": len(y),
            "Events": "—",
            "R²": r2_score(y, pred),
            "MAE": mean_absolute_error(y, pred),
            "RMSE": rmse(y, pred),
            "AUROC": "—",
            "AUPRC": "—",
            "Brier score": "—"
        })

        prediction_frames.append(pd.DataFrame({
            "Outcome": outcome_name,
            "Prediction task": "Regression",
            "Model": model_name,
            "Row index": data.index,
            "Observed": y,
            "Predicted": pred
        }))

    return rows, prediction_frames


def evaluate_classification_outcome(outcome_name):
    rows = []
    prediction_frames = []

    for model_name, features in MODEL_FEATURES.items():
        data = analysis[[outcome_name] + features].copy()
        data = data.dropna(subset=[outcome_name])

        y = data[outcome_name].astype(int).values
        X = data[features]

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        prob = cross_val_predict(
            make_classifier(),
            X,
            y,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        rows.append({
            "Outcome": outcome_name,
            "Prediction task": "Classification",
            "Model": model_name,
            "n": len(y),
            "Events": int(np.sum(y)),
            "R²": "—",
            "MAE": "—",
            "RMSE": "—",
            "AUROC": roc_auc_score(y, prob),
            "AUPRC": average_precision_score(y, prob),
            "Brier score": brier_score_loss(y, prob)
        })

        prediction_frames.append(pd.DataFrame({
            "Outcome": outcome_name,
            "Prediction task": "Classification",
            "Model": model_name,
            "Row index": data.index,
            "Observed": y,
            "Predicted probability": prob
        }))

    return rows, prediction_frames


all_rows = []
all_prediction_frames = []

for outcome in ["UFR (mL/min)", "SFR (mL/min)"]:
    rows, preds = evaluate_regression_outcome(outcome)
    all_rows.extend(rows)
    all_prediction_frames.extend(preds)

for outcome in ["UFR-defined hyposalivation", "SFR-defined hyposalivation"]:
    rows, preds = evaluate_classification_outcome(outcome)
    all_rows.extend(rows)
    all_prediction_frames.extend(preds)

model_performance = pd.DataFrame(all_rows)
predictions = pd.concat(all_prediction_frames, ignore_index=True)

model_performance_rounded = model_performance.copy()

for col in ["R²", "MAE", "RMSE", "AUROC", "AUPRC", "Brier score"]:
    model_performance_rounded[col] = model_performance_rounded[col].apply(
        lambda x: x if isinstance(x, str) else round(float(x), 3)
    )

print("\nTable 4. Model performance")
print(model_performance_rounded.to_string(index=False))


# ============================================================
# 7. DeLong comparisons for classification outcomes
# ============================================================

def get_prediction(outcome_name, model_name):
    sub = predictions[
        (predictions["Outcome"] == outcome_name) &
        (predictions["Model"] == model_name)
    ].copy()

    return sub.sort_values("Row index")


comparison_pairs = [
    (
        "Model 2: Systemic-medication burden model",
        "Model 1: Demographic model"
    ),
    (
        "Model 3: Practical oral-systemic model",
        "Model 2: Systemic-medication burden model"
    ),
    (
        "Model 4: Full salivary oral-systemic model",
        "Model 3: Practical oral-systemic model"
    ),
    (
        "Model 4: Full salivary oral-systemic model",
        "Model 1: Demographic model"
    )
]

delong_rows = []

for outcome_name in [
    "UFR-defined hyposalivation",
    "SFR-defined hyposalivation"
]:
    for model_a, model_b in comparison_pairs:
        pa = get_prediction(outcome_name, model_a)
        pb = get_prediction(outcome_name, model_b)

        merged = pa[
            ["Row index", "Observed", "Predicted probability"]
        ].merge(
            pb[["Row index", "Predicted probability"]],
            on="Row index",
            suffixes=(" A", " B")
        )

        y = merged["Observed"].astype(int).values
        prob_a = merged["Predicted probability A"].values
        prob_b = merged["Predicted probability B"].values

        auc_a, auc_b, p = delong_roc_test(y, prob_a, prob_b)

        delong_rows.append({
            "Outcome": outcome_name,
            "Comparison": f"{model_a} vs {model_b}",
            "AUROC, latter model": auc_a,
            "AUROC, former model": auc_b,
            "ΔAUROC": auc_a - auc_b,
            "DeLong p-value": p
        })

delong_table = pd.DataFrame(delong_rows)

print("\nSupplementary DeLong comparisons")
print(delong_table.to_string(index=False))


# ============================================================
# 8. ROC figures for classification outcomes
# ============================================================

def plot_roc_for_outcome(outcome_name, out_path):
    plt.figure(figsize=(5.5, 5.0))

    for model_name in MODEL_FEATURES.keys():
        sub = get_prediction(outcome_name, model_name)

        y = sub["Observed"].astype(int).values
        prob = sub["Predicted probability"].values

        fpr, tpr, _ = roc_curve(y, prob)
        auc_value = roc_auc_score(y, prob)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name.split(':')[0]} (AUROC={auc_value:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


plot_roc_for_outcome(
    "UFR-defined hyposalivation",
    OUT_DIR / "ROC_UFR_defined_hyposalivation.png"
)

plot_roc_for_outcome(
    "SFR-defined hyposalivation",
    OUT_DIR / "ROC_SFR_defined_hyposalivation.png"
)


# ============================================================
# 9. SHAP analysis for Model 4 classification models
# ============================================================

def get_shap_values_for_outcome(outcome_name, features):
    data = analysis[[outcome_name] + features].copy()
    data = data.dropna(subset=[outcome_name])

    y = data[outcome_name].astype(int).values
    X = data[features]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    X_imp_df = pd.DataFrame(
        X_imp,
        columns=features,
        index=data.index
    )

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_imp_df, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp_df)

    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    return y, X_imp_df, model, sv


def shap_importance_table(shap_values, feature_names, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)

    abs_sv = np.abs(shap_values)
    mean_abs = abs_sv.mean(axis=0)

    rows = []
    n = abs_sv.shape[0]

    for j, feature in enumerate(feature_names):
        boot_means = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            boot_means.append(abs_sv[idx, j].mean())

        low, high = np.percentile(boot_means, [2.5, 97.5])

        rows.append({
            "Predictor": feature,
            "Mean absolute SHAP": mean_abs[j],
            "Bootstrap 95% CI low": low,
            "Bootstrap 95% CI high": high,
            "Mean absolute SHAP, mean (95% bootstrap CI)": format_ci(
                mean_abs[j],
                low,
                high,
                digits=4
            )
        })

    out = (
        pd.DataFrame(rows)
        .sort_values("Mean absolute SHAP", ascending=False)
        .reset_index(drop=True)
    )

    out.insert(0, "Rank", np.arange(1, len(out) + 1))

    return out


model4_features = MODEL_FEATURES[
    "Model 4: Full salivary oral-systemic model"
]

shap_tables = []
compact_tables = []

for outcome_name in [
    "UFR-defined hyposalivation",
    "SFR-defined hyposalivation"
]:
    print(f"\nRunning SHAP for {outcome_name}...")

    y, X_imp_df, fitted_model, sv = get_shap_values_for_outcome(
        outcome_name,
        model4_features
    )

    imp = shap_importance_table(
        sv,
        X_imp_df.columns.tolist(),
        n_bootstrap=N_BOOTSTRAP,
        seed=RANDOM_STATE
    )

    imp.insert(0, "Outcome", outcome_name)
    shap_tables.append(imp)

    safe_name = outcome_name.replace(" ", "_").replace("/", "_")

    plt.figure()
    shap.summary_plot(sv, X_imp_df, show=False, max_display=12)
    plt.savefig(
        OUT_DIR / f"SHAP_beeswarm_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure()
    shap.summary_plot(
        sv,
        X_imp_df,
        plot_type="bar",
        show=False,
        max_display=12
    )
    plt.savefig(
        OUT_DIR / f"SHAP_bar_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Exploratory cumulative SHAP-informed compact models
    ordered_features = imp["Predictor"].tolist()

    compact_rows = []
    compact_predictions = {}

    for k in range(1, min(10, len(ordered_features)) + 1):
        feats = ordered_features[:k]

        data = analysis[[outcome_name] + feats].copy()
        data = data.dropna(subset=[outcome_name])

        yy = data[outcome_name].astype(int).values
        XX = data[feats]

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        prob = cross_val_predict(
            make_classifier(),
            XX,
            yy,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        auc = roc_auc_score(yy, prob)
        low, high = bootstrap_auc_ci(
            yy,
            prob,
            n_bootstrap=N_BOOTSTRAP,
            seed=RANDOM_STATE + k
        )

        compact_predictions[k] = (yy, prob)

        compact_rows.append({
            "Outcome": outcome_name,
            "Model": f"Top {k}",
            "Predictors": "; ".join(feats),
            "n": len(yy),
            "Events": int(np.sum(yy)),
            "AUROC": auc,
            "AUROC 95% CI low": low,
            "AUROC 95% CI high": high,
            "AUPRC": average_precision_score(yy, prob),
            "Brier score": brier_score_loss(yy, prob)
        })

    compact_df = pd.DataFrame(compact_rows)

    best_idx = compact_df["AUROC"].idxmax()
    best_k = int(compact_df.loc[best_idx, "Model"].replace("Top ", ""))

    best_y, best_prob = compact_predictions[best_k]

    p_values = []

    for _, row in compact_df.iterrows():
        k = int(row["Model"].replace("Top ", ""))
        y_k, prob_k = compact_predictions[k]

        if k == best_k:
            p_values.append(np.nan)
        else:
            _, _, p = delong_roc_test(y_k, prob_k, best_prob)
            p_values.append(p)

    compact_df["Post-hoc DeLong p vs best"] = p_values

    mask = compact_df["Post-hoc DeLong p vs best"].notna()
    compact_df["FDR-adjusted p vs best"] = np.nan

    if mask.sum() > 0:
        compact_df.loc[mask, "FDR-adjusted p vs best"] = multipletests(
            compact_df.loc[mask, "Post-hoc DeLong p vs best"],
            method="fdr_bh"
        )[1]

    compact_tables.append(compact_df)

shap_importance_all = pd.concat(shap_tables, ignore_index=True)
compact_models_all = pd.concat(compact_tables, ignore_index=True)


# ============================================================
# 10. Save outputs
# ============================================================

model_performance.to_csv(
    OUT_DIR / "Table4_model_performance_raw.csv",
    index=False,
    encoding="utf-8-sig"
)

model_performance_rounded.to_csv(
    OUT_DIR / "Table4_model_performance_rounded.csv",
    index=False,
    encoding="utf-8-sig"
)

predictions.to_csv(
    OUT_DIR / "Cross_validated_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

delong_table.to_csv(
    OUT_DIR / "Supplementary_Table_S2_DeLong_comparisons.csv",
    index=False,
    encoding="utf-8-sig"
)

shap_importance_all.to_csv(
    OUT_DIR / "Table5_SHAP_feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

compact_models_all.to_csv(
    OUT_DIR / "Supplementary_Table_S4_SHAP_compact_models.csv",
    index=False,
    encoding="utf-8-sig"
)

with pd.ExcelWriter(
    OUT_DIR / "ML_SHAP_results_workbook.xlsx",
    engine="openpyxl"
) as writer:
    model_performance_rounded.to_excel(
        writer,
        sheet_name="Table4_rounded",
        index=False
    )

    model_performance.to_excel(
        writer,
        sheet_name="Table4_raw",
        index=False
    )

    delong_table.to_excel(
        writer,
        sheet_name="DeLong",
        index=False
    )

    shap_importance_all.to_excel(
        writer,
        sheet_name="Table5_SHAP",
        index=False
    )

    compact_models_all.to_excel(
        writer,
        sheet_name="Compact_models_S4",
        index=False
    )

    predictions.to_excel(
        writer,
        sheet_name="CV_predictions",
        index=False
    )

print("\nSaved outputs to:")
print(OUT_DIR)

print("\nMain files:")
print("- ML_SHAP_results_workbook.xlsx")
print("- Table4_model_performance_rounded.csv")
print("- Supplementary_Table_S2_DeLong_comparisons.csv")
print("- Table5_SHAP_feature_importance.csv")
print("- Supplementary_Table_S4_SHAP_compact_models.csv")
print("- ROC_UFR_defined_hyposalivation.png")
print("- ROC_SFR_defined_hyposalivation.png")
print("- SHAP_beeswarm_*.png")
print("- SHAP_bar_*.png")