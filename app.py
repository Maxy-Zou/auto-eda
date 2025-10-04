import io
import pandas as pd
import streamlit as st

from utils import (
    infer_column_types,
    load_sample_dataset,
    safe_sample,
    build_report_html,
    fig_to_png_base64,
)
from eda import (
    render_univariate,
    render_missingness,
    render_correlation_heatmap,
    render_target_overview,
)
from model import (
    detect_task_type,
    train_baselines,
    render_classification_plots,
    render_regression_plots,
    best_model_from_metrics,
)

st.set_page_config(page_title="Auto-EDA + Model Starter", layout="wide")

# --- Cache helpers ---


@st.cache_data(show_spinner=False)
def _read_csv(file, **kwargs):
    return pd.read_csv(file, **kwargs)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Auto-EDA + Model Starter")
    st.info("Built by Max Zou. Upload a CSV → explore → model → explain.")
    st.caption("Upload a CSV or try a sample dataset. Explore EDA → train baselines → export a quick report.")
    sample = st.selectbox("Sample dataset", ["(none)", "Iris", "Titanic", "California Housing"], index=0)
    theme = st.radio("Theme", ["Minimal", "Neon", "Terminal"], index=0)
    st.divider()
    st.caption("Tip: Large datasets are auto-sampled for plots to keep things snappy.")


# --- Home tab: upload / load data ---
tab_home, tab_eda, tab_model = st.tabs(["Home", "EDA", "Model"])

with tab_home:
    st.header("Data")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    df = None

    if uploaded is not None:
        # Try a few robust options
        try:
            df = _read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            try:
                df = _read_csv(uploaded, sep=";")
            except Exception:
                uploaded.seek(0)
                df = _read_csv(uploaded, encoding_errors="ignore")

    if df is None and sample != "(none)":
        df = load_sample_dataset(sample)

    if df is None:
        st.info("Upload a CSV or choose a sample dataset from the sidebar.")
    else:
        st.success(f"Loaded data with shape {df.shape[0]} rows × {df.shape[1]} columns.")
        st.dataframe(df.head(20), use_container_width=True)
        coltypes = infer_column_types(df)
        with st.expander("Inferred column types"):
            st.json(coltypes)

        st.write("### Summary statistics")
        st.write(df.describe(include='all').transpose())

        st.session_state["__DATAFRAME__"] = df
        st.session_state["__COLTYPES__"] = coltypes


with tab_eda:
    st.header("Exploratory Data Analysis")
    if "__DATAFRAME__" not in st.session_state:
        st.warning("Load data in the **Home** tab first.")
    else:
        df = st.session_state["__DATAFRAME__"]
        coltypes = st.session_state["__COLTYPES__"]
        sampled = safe_sample(df)

        # Target selection (optional in EDA tab)
        target = st.selectbox("Optional target column", [
                              "(none)"] + list(df.columns), index=0)
        target = None if target == "(none)" else target

        c1, c2 = st.columns([2, 1])
        with c1:
            render_univariate(sampled, coltypes)
        with c2:
            render_missingness(df)

        render_correlation_heatmap(sampled, coltypes)

        if target:
            render_target_overview(df, target, coltypes)

with tab_model:
    st.header("Baseline Modeling")
    if "__DATAFRAME__" not in st.session_state:
        st.warning("Load data in the **Home** tab first.")
    else:
        df = st.session_state["__DATAFRAME__"]
        coltypes = st.session_state["__COLTYPES__"]

        # Choose target
        target = st.selectbox("Select target column", list(df.columns))
        task = detect_task_type(df, target, coltypes)
        st.write(f"**Detected task:** {task.capitalize()}")

        run = st.button("Train baselines")
        if run:
            with st.spinner("Training baselines..."):
                results = train_baselines(df, target, task, coltypes)

            st.subheader("Model Comparison")
            st.dataframe(results["comparison_table"], use_container_width=True)

            # Plots
            if task == "classification":
                render_classification_plots(results)
            else:
                render_regression_plots(results)

            # Feature importances (Random Forest if available)
            imp = None
            if task == "classification":
                imp = results["importances"].get("RandomForestClassifier")
            else:
                imp = results["importances"].get("RandomForestRegressor")
            if imp is not None:
                import pandas as pd
                st.subheader("Feature importances (Random Forest)")
                st.bar_chart(pd.Series(imp))

            # Export best
            best_name = best_model_from_metrics(results["metrics"], task)
            st.success(f"Best model: **{best_name}**")

            # Build HTML report
            st.subheader("Export")
            # Collect 2–3 key plot images from results (base64 PNGs)
            key_plots_b64 = []
            for fig in results.get("figs_for_report", []):
                key_plots_b64.append(fig_to_png_base64(fig))

            # Download all trained models
            with st.expander("Download all trained models"):
                for name, blob in results["model_bytes"].items():
                    st.download_button(
                        f"Download {name} (.pkl)",
                        data=blob,
                        file_name=f"{name}.pkl",
                        mime="application/octet-stream",
                        key=f"dl_{name}"
                    )


            report_html = build_report_html(
                dataset_summary={
                    "shape": f"{df.shape[0]} rows × {df.shape[1]} cols",
                    "target": target,
                    "task": task,
                },
                eda_highlights={
                    "numeric_count": len(coltypes["numeric"]),
                    "categorical_count": len(coltypes["categorical"]),
                    "datetime_count": len(coltypes["datetime"]),
                    "text_count": len(coltypes["text"]),
                },
                best_model_name=best_name,
                metrics=results["metrics"][best_name],
                key_plots_base64=key_plots_b64[:3],
            )

            report_bytes = report_html.encode("utf-8")
            st.download_button(
                "Download one-page HTML report",
                data=report_bytes,
                file_name="auto_eda_report.html",
                mime="text/html",
            )

            # Serialize model
            st.download_button(
                "Download best model (.pkl)",
                data=results["model_bytes"][best_name],
                file_name=f"{best_name}_model.pkl",
                mime="application/octet-stream",
            )
