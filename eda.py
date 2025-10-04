from typing import Dict, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def render_univariate(df: pd.DataFrame, coltypes: Dict[str, List[str]]):
    st.subheader("Univariate summaries")
    num_cols = coltypes["numeric"]
    cat_cols = coltypes["categorical"]

    if num_cols:
        st.markdown("**Numeric**")
        ncols = min(3, len(num_cols))
        for i, col in enumerate(num_cols):
            if i % ncols == 0:
                cols = st.columns(ncols)
            with cols[i % ncols]:
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=30)
                ax.set_title(col)
                st.pyplot(fig, use_container_width=True)

    if cat_cols:
        st.markdown("**Categorical**")
        ncols = min(2, len(cat_cols))
        for i, col in enumerate(cat_cols):
            if i % ncols == 0:
                cols = st.columns(ncols)
            with cols[i % ncols]:
                vc = df[col].astype(str).value_counts().head(20)
                fig, ax = plt.subplots()
                vc.plot(kind="bar", ax=ax)
                ax.set_title(col)
                ax.tick_params(axis="x", labelrotation=90)
                st.pyplot(fig, use_container_width=True)

def render_missingness(df: pd.DataFrame):
    st.subheader("Missing values")
    miss = df.isna().mean().sort_values(ascending=False)
    if miss.max() == 0:
        st.caption("No missing values detected.")
        return
    fig, ax = plt.subplots()
    miss.head(30).plot(kind="bar", ax=ax)
    ax.set_ylabel("Fraction missing")
    ax.tick_params(axis="x", labelrotation=90)
    st.pyplot(fig, use_container_width=True)

def render_correlation_heatmap(df: pd.DataFrame, coltypes: Dict[str, List[str]]):
    st.subheader("Correlation (numeric)")
    num_cols = coltypes["numeric"]
    if len(num_cols) < 2:
        st.caption("Not enough numeric columns for a correlation heatmap.")
        return
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax)
    ax.set_title("Pearson correlation")
    st.pyplot(fig, use_container_width=True)

def render_target_overview(df: pd.DataFrame, target: str, coltypes: Dict[str, List[str]]):
    st.subheader("Target overview")
    if target in coltypes["numeric"]:
        fig, ax = plt.subplots()
        ax.hist(df[target].dropna(), bins=30)
        ax.set_title(f"Distribution of {target}")
        st.pyplot(fig, use_container_width=True)
    else:
        vc = df[target].astype(str).value_counts()
        fig, ax = plt.subplots()
        vc.plot(kind="bar", ax=ax)
        ax.set_title(f"Counts of {target}")
        ax.tick_params(axis="x", labelrotation=90)
        st.pyplot(fig, use_container_width=True)
