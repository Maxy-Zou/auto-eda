from typing import Dict, List, Tuple
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,
    r2_score, mean_absolute_error, mean_squared_error
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils import bytes_from_model

def detect_task_type(df: pd.DataFrame, target: str, coltypes: Dict[str, List[str]]) -> str:
    """Return 'classification' or 'regression'."""
    if target in coltypes["numeric"]:
        # low unique numeric could still be classification, but we keep it simple
        if df[target].nunique(dropna=True) <= max(10, int(0.05 * len(df))):
            return "classification"
        return "regression"
    return "classification"

def _split_Xy(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def _build_preprocess(coltypes: Dict[str, List[str]]):
    num = coltypes["numeric"]
    cat = coltypes["categorical"]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, [c for c in num if c not in []]),
            ("cat", cat_pipe, [c for c in cat if c not in []]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre

def _train_classification(X_train, X_test, y_train, y_test, pre):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
    }
    metrics = {}
    trained = {}
    importances = {}

    for name, base in models.items():
        pipe = Pipeline([("pre", pre), ("est", base)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        m = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
        }

        # ROC-AUC (binary only)
        try:
            if len(np.unique(y_test)) == 2:
                proba = pipe.predict_proba(X_test)[:, 1]
                m["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            pass

        metrics[name] = m
        trained[name] = pipe

        # Feature importances (RF only)
        if name == "RandomForestClassifier":
            try:
                est = pipe.named_steps["est"]
                importances[name] = est.feature_importances_
            except Exception:
                importances[name] = None

    return trained, metrics, importances

def _train_regression(X_train, X_test, y_train, y_test, pre):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
    }
    metrics = {}
    trained = {}
    importances = {}

    for name, base in models.items():
        pipe = Pipeline([("pre", pre), ("est", base)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        m = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
        }
        metrics[name] = m
        trained[name] = pipe

        if name == "RandomForestRegressor":
            try:
                est = pipe.named_steps["est"]
                importances[name] = est.feature_importances_
            except Exception:
                importances[name] = None

    return trained, metrics, importances

def train_baselines(df: pd.DataFrame, target: str, task: str, coltypes: Dict[str, List[str]]):
    X, y = _split_Xy(df, target)
    # Ensure target is suitable for classification
    stratify = y if (task == "classification" and y.nunique() > 1) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    pre = _build_preprocess(coltypes)

    if task == "classification":
        trained, metrics, importances = _train_classification(X_train, X_test, y_train, y_test, pre)
    else:
        trained, metrics, importances = _train_regression(X_train, X_test, y_train, y_test, pre)

    # Build comparison table
    rows = []
    if task == "classification":
        for k, m in metrics.items():
            rows.append({"model": k, "accuracy": m.get("accuracy"), "f1": m.get("f1"), "roc_auc": m.get("roc_auc")})
        cmp_df = pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False, na_position="last")
    else:
        for k, m in metrics.items():
            rows.append({"model": k, "r2": m.get("r2"), "mae": m.get("mae"), "rmse": m.get("rmse")})
        cmp_df = pd.DataFrame(rows).sort_values(["r2"], ascending=False)

    # Serialize models to bytes for download
    model_bytes = {name: bytes_from_model(model) for name, model in trained.items()}

    # Prepare some plot figs for report
    figs_for_report = []
    if task == "classification":
        figs_for_report.extend(_make_classification_report_figs(trained, X_test, y_test))
    else:
        figs_for_report.extend(_make_regression_report_figs(trained, X_test, y_test))

    return {
        "trained": trained,
        "metrics": metrics,
        "comparison_table": cmp_df.reset_index(drop=True),
        "importances": importances,
        "figs_for_report": figs_for_report[:3],
        "model_bytes": model_bytes,
    }

def best_model_from_metrics(metrics: dict, task: str) -> str:
    if task == "classification":
        # prioritize F1, fall back to accuracy
        best = max(metrics.items(), key=lambda kv: (kv[1].get("f1", -1), kv[1].get("accuracy", -1)))
        return best[0]
    else:
        best = max(metrics.items(), key=lambda kv: (kv[1].get("r2", -1), -kv[1].get("rmse", 1e9)))
        return best[0]

# ---------- Plots rendered in Streamlit ----------

def render_classification_plots(results: dict):
    st.subheader("Classification diagnostics")
    # Pick any one trained model for visuals (best is fine)
    name = best_model_from_metrics(results["metrics"], "classification")
    model = results["trained"][name]
    # We don't have X_test,y_test saved; reconstruct via impossible here—so instead,
    # we created report figs during training. Show any report figs if present.
    if results.get("figs_for_report"):
        for fig in results["figs_for_report"]:
            st.pyplot(fig, use_container_width=True)
    else:
        st.caption("No diagnostic plots were generated.")

def render_regression_plots(results: dict):
    st.subheader("Regression diagnostics")
    if results.get("figs_for_report"):
        for fig in results["figs_for_report"]:
            st.pyplot(fig, use_container_width=True)
    else:
        st.caption("No diagnostic plots were generated.")

# ---------- Internal helpers to make figs we can both show & export ----------

def _make_classification_report_figs(trained: dict, X_test, y_test):
    figs = []
    # Use RF if present for ROC/CM; else first model
    model = trained.get("RandomForestClassifier", next(iter(trained.values())))
    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    fig_cm, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(np.unique(y_test))))
    ax.set_yticks(range(len(np.unique(y_test))))
    ax.set_xticklabels(np.unique(y_test).astype(str), rotation=45)
    ax.set_yticklabels(np.unique(y_test).astype(str))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    figs.append(fig_cm)

    # ROC for binary
    if len(np.unique(y_test)) == 2:
        try:
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            fig_roc, ax = plt.subplots()
            ax.plot(fpr, tpr, label="ROC")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title("ROC Curve (RF)")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            figs.append(fig_roc)
        except Exception:
            pass

    return figs

def _make_regression_report_figs(trained: dict, X_test, y_test):
    figs = []
    model = trained.get("RandomForestRegressor", next(iter(trained.values())))
    y_pred = model.predict(X_test)

    # Residuals vs fitted
    resid = y_test - y_pred
    fig_sc, ax = plt.subplots()
    ax.scatter(y_pred, resid, s=10, alpha=0.7)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    figs.append(fig_sc)

    # Residual histogram
    fig_hist, ax = plt.subplots()
    ax.hist(resid, bins=30)
    ax.set_title("Residuals histogram")
    figs.append(fig_hist)

    return figs
