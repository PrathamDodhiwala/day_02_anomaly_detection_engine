import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_utils import load_data, prepare_features, build_demo_data
from src.detectors import run_isolation_forest, run_dbscan
from src.metrics import summarize_results
from src.ui import apply_theme

st.set_page_config(page_title="Anomaly Detection Engine", page_icon="◈", layout="wide")
apply_theme()
if "data" not in st.session_state: st.session_state.data = None
if "results" not in st.session_state: st.session_state.results = None

with st.sidebar:
    st.markdown("## ◈ A D E")
    st.caption("ANOMALY DETECTION ENGINE")
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "Dataset", "Detection Lab", "Results & Export", "About"])
    st.markdown("---")
    st.caption("DAY 02 / 60-DAY PROJECT MARATHON")

st.markdown('<div class="eyebrow">INTELLIGENCE WORKSPACE / DAY 02</div>', unsafe_allow_html=True)
st.title("Anomaly Detection Engine")
st.markdown("Identify unusual patterns in structured data using unsupervised machine learning.")

if page == "Overview":
    st.markdown("### Detection workspace")
    df, res = st.session_state.data, st.session_state.results
    cols = st.columns(4)
    metrics = [
        ("DATASET", f"{len(df):,}" if df is not None else "—", "Rows loaded"),
        ("FEATURES", str(df.select_dtypes(include="number").shape[1]) if df is not None else "—", "Numeric columns"),
        ("MODEL STATUS", "READY" if df is not None else "WAITING", "Load data to begin"),
        ("ANOMALIES", f"{int(res['is_anomaly'].sum()):,}" if res is not None else "—", "Latest run")
    ]
    for c, (label, value, help_text) in zip(cols, metrics):
        c.metric(label, value, help=help_text)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Workflow")
        for n, title, desc in [("01","Load data","Upload a CSV or use the demo dataset."),
                               ("02","Prepare features","Select numeric columns and standardize them."),
                               ("03","Run detection","Compare Isolation Forest and DBSCAN."),
                               ("04","Investigate","Review flagged records and export results.")]:
            st.markdown(f'<div class="step"><span>{n}</span><div><b>{title}</b><small>{desc}</small></div></div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Algorithms")
        st.markdown('<div class="info-card"><b>Isolation Forest</b><p>Uses random feature splits to isolate observations. Shorter paths can indicate anomalies.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card"><b>DBSCAN</b><p>Finds dense clusters and labels points outside dense regions as noise.</p></div>', unsafe_allow_html=True)
        st.info("An anomaly flag indicates an unusual pattern, not proof of fraud or error.")

elif page == "Dataset":
    st.markdown("### Dataset management")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load uploaded dataset", type="primary", disabled=uploaded is None):
            try:
                st.session_state.data = load_data(uploaded)
                st.session_state.results = None
                st.success(f"Loaded {len(st.session_state.data):,} rows.")
            except Exception as e:
                st.error(f"Could not load file: {e}")
    with c2:
        if st.button("Use generated demo dataset"):
            st.session_state.data = build_demo_data()
            st.session_state.results = None
            st.success("Demo dataset loaded.")
    if st.session_state.data is not None:
        df = st.session_state.data
        st.dataframe(df.head(100), use_container_width=True)
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", len(df.columns))
        c.metric("Missing cells", int(df.isna().sum().sum()))
        profile = pd.DataFrame({"Column": df.columns, "Type": [str(df[x].dtype) for x in df.columns],
                                "Missing": [int(df[x].isna().sum()) for x in df.columns],
                                "Unique": [int(df[x].nunique()) for x in df.columns]})
        st.markdown("#### Column profile")
        st.dataframe(profile, use_container_width=True, hide_index=True)
    else:
        st.info("Upload a CSV or load the demo dataset to begin.")

elif page == "Detection Lab":
    st.markdown("### Detection lab")
    df = st.session_state.data
    if df is None:
        st.warning("Load a dataset first in Dataset.")
    else:
        numeric = df.select_dtypes(include="number").columns.tolist()
        if not numeric:
            st.error("No numeric columns found. At least one numeric feature is required.")
        else:
            selected = st.multiselect("Numeric features", numeric, default=numeric[:min(5, len(numeric))])
            a, b = st.columns(2)
            with a:
                st.markdown("#### Isolation Forest")
                contamination = st.slider("Expected anomaly proportion", .01, .30, .05, .01)
                use_if = st.checkbox("Run Isolation Forest", True)
            with b:
                st.markdown("#### DBSCAN")
                eps = st.slider("Epsilon (neighbourhood radius)", .1, 5.0, .8, .1)
                min_samples = st.slider("Minimum samples", 2, 30, 5)
                use_db = st.checkbox("Run DBSCAN", True)
            if st.button("Run anomaly detection", type="primary", disabled=not selected or not (use_if or use_db)):
                try:
                    X, clean = prepare_features(df, selected)
                    result = clean.copy()
                    if use_if:
                        out = run_isolation_forest(X, contamination=contamination)
                        for col in out: result[col] = out[col].values
                    if use_db:
                        out = run_dbscan(X, eps=eps, min_samples=min_samples)
                        for col in out: result[col] = out[col].values
                    detector_cols = [x for x in result if x.endswith("_anomaly")]
                    result["is_anomaly"] = result[detector_cols].any(axis=1)
                    st.session_state.results = result
                    st.success(f"Detection complete. Analysed {len(result):,} records.")
                except Exception as e:
                    st.error(f"Detection failed: {e}")
            if st.session_state.results is not None:
                st.markdown("#### Latest run preview")
                st.dataframe(st.session_state.results.head(100), use_container_width=True)
                st.caption("Rows missing selected feature values are excluded during preprocessing.")

elif page == "Results & Export":
    st.markdown("### Results & investigation")
    res = st.session_state.results
    if res is None:
        st.info("Run a detector in the Detection Lab first.")
    else:
        summary = summarize_results(res)
        a, b, c = st.columns(3)
        a.metric("Records analysed", f"{summary['total']:,}")
        b.metric("Flagged anomalies", f"{summary['anomalies']:,}")
        c.metric("Anomaly rate", f"{summary['rate']:.1%}")
        detector_cols = [x for x in res if x.endswith("_anomaly")]
        if detector_cols:
            counts = pd.DataFrame({"Detector": [x.replace("_anomaly", "").replace("_", " ").title() for x in detector_cols],
                                   "Anomalies": [int(res[x].sum()) for x in detector_cols]})
            st.plotly_chart(px.bar(counts, x="Detector", y="Anomalies", text="Anomalies",
                                   title="Anomalies by detector"), use_container_width=True)
        numeric = res.select_dtypes(include="number").columns.tolist()
        if numeric:
            plot_df = res.copy()
            plot_df["Detection status"] = plot_df["is_anomaly"].map({True: "Anomaly", False: "Normal"})
            xcol = numeric[0]
            ycol = numeric[1] if len(numeric) > 1 else numeric[0]
            st.plotly_chart(px.scatter(plot_df, x=xcol, y=ycol, color="Detection status",
                                       title="Flagged vs normal observations"), use_container_width=True)
        st.markdown("#### Flagged records")
        flagged = res[res["is_anomaly"]]
        st.dataframe(flagged, use_container_width=True)
        st.download_button("Download complete results (CSV)", res.to_csv(index=False).encode(),
                           "anomaly_detection_results.csv", "text/csv", type="primary")
        st.download_button("Download flagged records (CSV)", flagged.to_csv(index=False).encode(),
                           "flagged_anomalies.csv", "text/csv")
else:
    st.markdown("### About this project")
    st.markdown("A reusable unsupervised-learning application built with Python, Streamlit, Pandas, Plotly and Scikit-learn.")
    st.markdown("**Learning objectives**")
    st.write("- Build a data ingestion and preprocessing workflow.\n- Apply Isolation Forest and DBSCAN without labelled outcomes.\n- Compare detector outputs and investigate unusual observations.\n- Communicate results with interactive visualisations.")
    st.warning("Detector results depend on feature selection, scaling and parameter settings. Validate flagged observations with domain knowledge.")
    st.markdown("**Stack:** Python 3.10 · Streamlit · Pandas · NumPy · Scikit-learn · Plotly · pytest")
