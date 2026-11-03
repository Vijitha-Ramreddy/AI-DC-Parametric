from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


DEFAULT_API = os.getenv("DCOPT_API_URL", "http://localhost:8000")
UPLOAD_TESTS = [
    ("continuity", "Continuity Test"),
    ("leakage", "Leakage Test"),
    ("output_voltage", "Output Voltage Test"),
    ("power_consumption", "Power Consumption Test"),
    ("supply_voltage", "Supply Voltage Test"),
]
STAGE_NAMES = {
    "io": "Data Quality",
    "phase0_data_foundation": "Data Foundation",
    "01_baseline": "Baseline Analysis",
    "02_settling": "Settling Analysis",
    "03_static": "Static Test Reduction",
    "04_ordering": "Test Ordering",
    "05_predictive": "Predictive Testing",
    "06_adaptive": "Adaptive Testing",
    "07_pat": "Production Adaptive Testing",
    "08_scorecard": "Optimization Scorecard",
    "09_monitoring": "Monitoring",
}

st.set_page_config(page_title="DC Test-Time Optimizer", layout="wide")

with st.sidebar:
    st.title("DC Optimizer")
    st.divider()
    page = st.radio("View", ["Overview", "Upload & Analyze", "Data quality", "Artifacts"])


def api_get(path: str, timeout: int = 30):
    response = requests.get(f"{DEFAULT_API}{path}", timeout=timeout)
    response.raise_for_status()
    return response


try:
    api_get("/api/health")
except requests.RequestException:
    st.error("FastAPI backend is unavailable. Start it with `uvicorn api:app --port 8000`.")
    st.stop()

st.title("AI-Optimized DC Parametric Test-Time Reduction")

if page == "Overview":
    config_response = api_get("/api/config")
    config = config_response.json()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DPPM budget", config["dppm_budget"])
    col2.metric("Target reduction", f"{config['target_reduction_pct']}%")
    col3.metric("Train lots", len(config["train_lots"]))
    col4.metric("Holdout lots", len(config["holdout_lots"]))

    quality = api_get("/api/data/quality").json()
    st.subheader("Dataset overview")
    st.dataframe(
        pd.DataFrame([
            {"Dataset": name, "Rows": rows}
            for name, rows in quality["row_counts"].items()
        ]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Lots: " + ", ".join(quality["lots"])
        + " | Testers: " + ", ".join(quality["testers"])
    )

elif page == "Upload & Analyze":
    st.header("Upload and Analyze Dataset")
    uploaded = st.file_uploader(
        "Upload a CSV or Excel dataset",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded is not None:
        if (
            st.session_state.get("uploaded_filename") != uploaded.name
            or st.session_state.get("upload_response") is None
        ):
            with st.spinner("Reading and validating uploaded dataset..."):
                response = requests.post(
                    f"{DEFAULT_API}/api/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=120,
                )
                if response.ok:
                    st.session_state.upload_response = response.json()
                    st.session_state.uploaded_filename = uploaded.name
                else:
                    st.error(response.json().get("detail", "Dataset upload failed"))

        upload = st.session_state.get("upload_response")
        if upload:
            col1, col2, col3 = st.columns(3)
            col1.metric("Records", upload["cleaning"]["rows"])
            col2.metric("Columns", upload["cleaning"]["columns"])
            col3.metric("Duplicate rows removed", upload["cleaning"]["duplicates_removed"])

            st.subheader("Dataset preview")
            st.dataframe(pd.DataFrame(upload["preview"]), use_container_width=True, hide_index=True)

            st.subheader("Column mapping")
            st.caption("Map each DC parameter to a column. Missing mappings will be reported clearly.")
            columns = [""] + upload["columns"]
            suggestions = upload["suggested_mappings"]
            mappings = {}
            mapping_columns = st.columns(2)
            for index, (key, title) in enumerate(UPLOAD_TESTS):
                suggested = suggestions.get(key)
                default_index = columns.index(suggested) if suggested in columns else 0
                mappings[key] = mapping_columns[index % 2].selectbox(
                    title,
                    columns,
                    index=default_index,
                    key=f"mapping_{key}",
                ) or None

            suggested_label = suggestions.get("label")
            label_default = columns.index(suggested_label) if suggested_label in columns else 0
            mappings["measurement_time"] = st.selectbox(
                "Measurement time column (optional)",
                columns,
                index=columns.index(suggestions.get("measurement_time"))
                if suggestions.get("measurement_time") in columns else 0,
                help="Use the original cumulative measurement time column, if available.",
            ) or None
            label_column = st.selectbox(
                "Pass/fail label column (optional)",
                columns,
                index=label_default,
            ) or None

            missing = [title for key, title in UPLOAD_TESTS if not mappings[key]]
            if missing:
                st.warning("Missing required mapped parameters: " + ", ".join(missing))

            if st.button("Analyze Dataset", type="primary", use_container_width=True):
                analysis_response = requests.post(
                    f"{DEFAULT_API}/api/upload/analyze",
                    json={
                        "upload_id": upload["upload_id"],
                        "mappings": mappings,
                        "label_column": label_column,
                    },
                    timeout=300,
                )
                if analysis_response.ok:
                    st.session_state.dynamic_analysis = analysis_response.json()
                    st.success("Dynamic analysis completed")
                else:
                    st.error(analysis_response.json().get("detail", "Analysis failed"))

    analysis = st.session_state.get("dynamic_analysis")
    if analysis:
        st.divider()
        st.header("Optimization Results")
        individual = pd.DataFrame(analysis["individual_tests"])
        cleaning = analysis["cleaning"]
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Records analyzed", cleaning["rows"])
        metric2.metric("Columns", cleaning["columns"])
        if analysis["measurement_efficiency"]["original"]:
            metric3.metric("Original time", f"{analysis['measurement_efficiency']['original']:,.2f}")
            metric4.metric(
                "Estimated optimized time",
                f"{analysis['measurement_efficiency']['optimized_estimated']:,.2f}",
            )
        else:
            metric3.metric("Original time", "Not available")
            metric4.metric("Optimized time", "Not available")

        efficiency = analysis["measurement_efficiency"]
        st.subheader("A. Measurement Efficiency")
        st.info(efficiency["basis"])
        if efficiency["original"]:
            fig = go.Figure(data=[
                go.Bar(name="Original", x=["Measurement time"], y=[efficiency["original"]], marker_color="#4C78A8"),
                go.Bar(
                    name="Optimized estimate",
                    x=["Measurement time"],
                    y=[efficiency["optimized_estimated"]],
                    marker_color="#59A14F",
                ),
            ])
            fig.update_layout(barmode="group", yaxis_title=efficiency["measurement_unit"])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("B. Static Test Reduction")
        static_df = pd.DataFrame(analysis["static_reduction"])
        st.dataframe(static_df, use_container_width=True, hide_index=True)
        kept = int(static_df["Decision"].ne("Potential review candidate").sum()) if not static_df.empty else 0
        candidate_count = len(static_df) - kept if not static_df.empty else 0
        fig = go.Figure(data=[
            go.Bar(name="Original tests", x=["Tests"], y=[len(static_df)], marker_color="#4C78A8"),
            go.Bar(name="Proposed retained tests", x=["Tests"], y=[kept], marker_color="#F28E2B"),
        ])
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Potential review candidates: {candidate_count}. Candidates require validation before removal.")

        st.subheader("C. Predictive Testing")
        predictive = analysis["predictive"]
        if not predictive.get("available"):
            st.warning(predictive["message"])
        else:
            metric_columns = st.columns(4)
            metric_columns[0].metric("Accuracy", f"{predictive['accuracy']:.3f}")
            metric_columns[1].metric("Precision", f"{predictive['precision']:.3f}")
            metric_columns[2].metric("Recall", f"{predictive['recall']:.3f}")
            metric_columns[3].metric("F1-score", f"{predictive['f1_score']:.3f}")
            matrix = predictive["confusion_matrix"]
            fig = go.Figure(go.Heatmap(
                z=matrix,
                x=["Predicted Pass", "Predicted Fail"],
                y=["Actual Pass", "Actual Fail"],
                colorscale="Blues",
                text=matrix,
                texttemplate="%{text}",
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            predictions = pd.DataFrame(analysis["predictive_predictions"])
            st.dataframe(predictions, use_container_width=True, hide_index=True)

        st.subheader("D. Adaptive Testing")
        adaptive_df = pd.DataFrame(analysis["adaptive"])
        st.dataframe(adaptive_df, use_container_width=True, hide_index=True)
        priority_order = {"MANDATORY": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        adaptive_df["_priority"] = adaptive_df["Priority"].map(priority_order)
        fig = go.Figure(go.Bar(
            x=adaptive_df["Recommended test"],
            y=adaptive_df["_priority"],
            text=adaptive_df["Priority"],
            textposition="auto",
            marker_color="#9D79BC",
        ))
        fig.update_layout(yaxis=dict(tickvals=[1, 2, 3, 4], ticktext=["LOW", "MEDIUM", "HIGH", "MANDATORY"]))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Individual DC Parametric Test Analysis")
        if individual.empty:
            st.warning("No test parameter columns were mapped.")
        else:
            st.dataframe(individual, use_container_width=True, hide_index=True)
            failures = individual[["Test", "Failing chips"]].copy()
            fig = go.Figure(go.Bar(
                x=failures["Test"],
                y=failures["Failing chips"],
                marker_color="#E15759",
            ))
            fig.update_layout(yaxis_title="Failing chips")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Download results")
        download_response = requests.get(
            f"{DEFAULT_API}{analysis['processed_download']}",
            timeout=120,
        )
        if download_response.ok:
            st.download_button(
                "Download processed dataset",
                download_response.content,
                file_name="processed_dataset.csv",
                mime="text/csv",
            )

        def flatten_downloads() -> pd.DataFrame:
            frames = []
            for name, payload in (
                ("individual_tests", analysis["individual_tests"]),
                ("static_reduction", analysis["static_reduction"]),
                ("adaptive", analysis["adaptive"]),
            ):
                frame = pd.DataFrame(payload)
                frame.insert(0, "result_type", name)
                frames.append(frame)
            return pd.concat(frames, ignore_index=True)

        st.download_button(
            "Download optimization results CSV",
            flatten_downloads().to_csv(index=False).encode("utf-8"),
            file_name="dynamic_optimization_results.csv",
            mime="text/csv",
        )
        report = [
            "# Dynamic DC Parametric Test Analysis",
            "",
            f"- Records analyzed: {cleaning['rows']}",
            f"- Columns analyzed: {cleaning['columns']}",
            f"- Measurement efficiency basis: {efficiency['basis']}",
            "",
            "## Missing required test mappings",
            "None" if not analysis["missing_tests"] else ", ".join(analysis["missing_tests"]),
            "",
            "## Predictive testing",
            predictive["message"] if not predictive.get("available") else
            f"Accuracy: {predictive['accuracy']:.3f}; Precision: {predictive['precision']:.3f}; "
            f"Recall: {predictive['recall']:.3f}; F1: {predictive['f1_score']:.3f}",
        ]
        st.download_button(
            "Download analysis report",
            "\n".join(report).encode("utf-8"),
            file_name="dynamic_analysis_report.md",
            mime="text/markdown",
        )

elif page == "Data quality":
    quality = api_get("/api/data/quality").json()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rows")
        st.dataframe(
            pd.DataFrame(
                [{"dataset": key, "rows": value} for key, value in quality["row_counts"].items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with col2:
        st.subheader("Missing values")
        nulls = quality["null_columns"]
        st.dataframe(
            pd.DataFrame([{"column": key, "null_count": value} for key, value in nulls.items()])
            if nulls else pd.DataFrame([{"status": "No null values"}]),
            hide_index=True,
            use_container_width=True,
        )
    st.subheader("Coverage")
    st.write({
        "Lots": quality["lots"],
        "Testers": quality["testers"],
        "Temperatures": quality["temperatures"],
    })

else:
    stages = api_get("/api/results").json()["stages"]
    if not stages:
        st.info("No existing project result artifacts are available.")
        st.stop()
    stage = st.selectbox(
        "Result stage",
        stages,
        format_func=lambda value: STAGE_NAMES.get(value, value),
    )
    files = api_get(f"/api/results/{stage}").json()["files"]
    image_files = [name for name in files if name.endswith(".png")]

    summary_name = "summary.md"
    if summary_name in files:
        summary = api_get(f"/api/results/{stage}/{summary_name}").json()["content"]
        st.subheader("Output summary")
        st.markdown(summary)

    if image_files:
        st.subheader("Charts")
        chart_columns = st.columns(min(2, len(image_files)))
        for index, image_name in enumerate(image_files):
            column = chart_columns[index % len(chart_columns)]
            image_response = api_get(f"/api/results/{stage}/{image_name}")
            column.image(image_response.content, caption=image_name, use_container_width=True)

    st.subheader("Browse generated artifacts")
    artifact = st.selectbox("Artifact", files)
    response = api_get(f"/api/results/{stage}/{artifact}")
    if artifact.endswith(".csv"):
        payload = response.json()
        st.caption(f"Showing {len(payload['rows'])} of {payload['total_rows']} rows")
        st.dataframe(pd.DataFrame(payload["rows"]), hide_index=True, use_container_width=True)
    elif artifact.endswith(".md"):
        st.markdown(response.json()["content"])
    else:
        st.image(response.content, use_column_width=True)
