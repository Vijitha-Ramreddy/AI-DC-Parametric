"""Dynamic analysis for datasets uploaded through the Streamlit dashboard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


CANONICAL_TESTS = {
    "continuity": "Continuity Test",
    "leakage": "Leakage Test",
    "output_voltage": "Output Voltage Test",
    "power_consumption": "Power Consumption Test",
    "supply_voltage": "Supply Voltage Test",
}

TEST_PATTERNS = {
    "continuity": ("continuity", "cont", "diode"),
    "leakage": ("leak", "leakage", "iil", "iih", "ios"),
    "output_voltage": ("output voltage", "output_voltage", "voh", "vol"),
    "power_consumption": ("power", "iddq", "idd_static", "idd static", "current consumption"),
    "supply_voltage": ("supply voltage", "supply_voltage", "vcc", "vih", "vil"),
}

LABEL_PATTERNS = ("sort_pass", "label", "pass_fail", "pass/fail", "result", "status", "fail")
TIME_PATTERNS = ("test_time", "measurement_time", "measure_time", "duration_ms", "time_ms")


def normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def read_uploaded(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Only CSV and Excel files are supported")


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    original_rows = len(frame)
    duplicate_count = int(frame.duplicated().sum())
    cleaned = frame.drop_duplicates().copy()

    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].replace(r"^\s*$", np.nan, regex=True)

    missing_before = int(cleaned.isna().sum().sum())
    numeric_columns: list[str] = []
    converted_columns: list[str] = []
    for column in cleaned.columns:
        series = cleaned[column]
        if series.dtype == object and series.notna().any():
            converted = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
            if converted.notna().sum() >= max(1, series.notna().sum() * 0.8):
                cleaned[column] = converted
                converted_columns.append(str(column))
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            numeric_columns.append(str(column))

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned, {
        "original_rows": original_rows,
        "rows": len(cleaned),
        "columns": len(cleaned.columns),
        "duplicates_removed": duplicate_count,
        "missing_values": missing_before,
        "numeric_columns_converted": converted_columns,
    }


def suggest_mappings(frame: pd.DataFrame) -> dict[str, str | None]:
    suggestions: dict[str, str | None] = {}
    columns = {normalized(column): str(column) for column in frame.columns}

    def find(patterns: tuple[str, ...]) -> str | None:
        for column_key, column in columns.items():
            if any(normalized(pattern) in column_key for pattern in patterns):
                if column_key.endswith("_pf"):
                    continue
                return column
        return None

    for key, patterns in TEST_PATTERNS.items():
        suggestions[key] = find(patterns)

    explicit_labels = ("sort_pass", "pass_fail", "pass_fail_label", "overall_pass_fail", "result")
    suggested_label = find(explicit_labels)
    suggestions["label"] = suggested_label or find(LABEL_PATTERNS)
    suggestions["measurement_time"] = find(TIME_PATTERNS)
    return suggestions


def normalize_status(series: pd.Series) -> pd.Series | None:
    text = series.astype("string").str.strip().str.upper()
    mapping = {
        "P": 1, "PASS": 1, "PASSED": 1, "GOOD": 1, "1": 1, "TRUE": 1, "YES": 1, "Y": 1,
        "F": 0, "FAIL": 0, "FAILED": 0, "BAD": 0, "0": 0, "FALSE": 0, "NO": 0, "N": 0,
    }
    values = text.map(mapping)
    if values.notna().any() and values.notna().mean() >= 0.8:
        return values.astype("float")
    return None


def infer_label(frame: pd.DataFrame, selected: str | None = None) -> tuple[pd.Series | None, str | None]:
    if selected and selected in frame.columns:
        status = normalize_status(frame[selected])
        if status is not None:
            return status, selected

    for pattern in LABEL_PATTERNS:
        matches = [
            column for column in frame.columns
            if pattern in normalized(column) and not normalized(column).endswith("_pf")
        ]
        for column in matches:
            status = normalize_status(frame[column])
            if status is not None:
                return status, column
    return None, None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def individual_test_analysis(
    frame: pd.DataFrame,
    mappings: dict[str, str | None],
    label: pd.Series | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for key, title in CANONICAL_TESTS.items():
        column = mappings.get(key)
        if not column or column not in frame.columns:
            missing.append(title)
            continue

        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        pass_mask = label
        pf_column = f"{column}_PF"
        if pass_mask is None and pf_column in frame.columns:
            pass_mask = normalize_status(frame[pf_column])

        if pass_mask is not None:
            pass_mask = pass_mask.reindex(frame.index)
            passing = int(pass_mask.eq(1).sum())
            failing = int(pass_mask.eq(0).sum())
            total = passing + failing
        else:
            total = len(values)
            passing = failing = 0

        opportunity = "Review specification limits and measurement stability."
        if total and failing / total >= 0.05:
            opportunity = "Prioritize for additional testing; current failure rate warrants validation."
        elif total == 0:
            opportunity = "Insufficient labeled pass/fail data for evidence-based prioritization."

        rows.append({
            "Test": title,
            "Mapped column": column,
            "Total tested chips": total,
            "Passing chips": passing,
            "Failing chips": failing,
            "Pass percentage": round(passing / total * 100, 3) if total else None,
            "Fail percentage": round(failing / total * 100, 3) if total else None,
            "Mean": safe_float(values.mean()),
            "Median": safe_float(values.median()),
            "Standard deviation": safe_float(values.std(ddof=1)),
            "Minimum": safe_float(values.min()),
            "Maximum": safe_float(values.max()),
            "Optimization opportunity": opportunity,
            "Safety critical": key == "supply_voltage",
        })
    return rows, missing


def static_reduction(
    frame: pd.DataFrame,
    mappings: dict[str, str | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    critical_names = ("voltage", "supply", "continuity", "safety")
    for key, title in CANONICAL_TESTS.items():
        column = mappings.get(key)
        if not column or column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        pf_column = f"{column}_PF"
        if pf_column in frame.columns:
            pf = normalize_status(frame[pf_column])
            failure_count = int(pf.eq(0).sum()) if pf is not None else 0
        else:
            failure_count = 0

        variance_zero = bool(values.notna().sum() > 1 and values.std(ddof=1) == 0)
        critical = bool(any(word in normalized(title) for word in critical_names))
        candidate = failure_count == 0 and variance_zero and not critical
        decision = "Potential review candidate" if candidate else "KEEP"
        reason = (
            "Zero measured variance and no observed failures; confirm specification and manufacturing history."
            if candidate else "No sufficient validation evidence; mandatory or safety-critical tests are retained."
        )
        rows.append({
            "Test": title,
            "Mapped column": column,
            "Observed failures": failure_count,
            "Decision": decision,
            "Reason": reason,
            "Safety critical": critical,
        })
    return rows


def predictive_analysis(
    frame: pd.DataFrame,
    label: pd.Series | None,
    label_column: str | None,
) -> dict[str, Any]:
    if label is None or label.nunique(dropna=True) < 2:
        return {
            "available": False,
            "message": "Supervised evaluation is unavailable because no valid binary failure/pass label was found.",
        }

    work = frame.copy()
    work["_target"] = label
    leakage_patterns = ("pf", "pass", "fail", "sort", "label", "result", "status", "bin", "hard_bin", "first_fail")
    feature_columns = [
        column for column in work.select_dtypes(include=[np.number]).columns
        if column != "_target"
        and normalized(column) not in {"sort_pass", "hard_bin"}
        and not any(pattern in normalized(column) for pattern in leakage_patterns)
    ]
    feature_columns = [column for column in feature_columns if work[column].notna().any()]
    valid = work["_target"].notna()
    if len(feature_columns) < 1 or int(valid.sum()) < 20:
        return {
            "available": False,
            "message": "Not enough numeric features or labeled records for a reliable train/test model.",
        }

    x = work.loc[valid, feature_columns]
    y = work.loc[valid, "_target"].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        return {
            "available": False,
            "message": "The labeled data does not contain at least two records in each pass/fail class.",
        }

    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(x_test))
    predictions = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predicted,
        "failure_probability": probabilities,
    })
    return {
        "available": True,
        "label_column": label_column,
        "feature_count": len(feature_columns),
        "test_records": len(y_test),
        "accuracy": safe_float(accuracy_score(y_test, predicted)),
        "precision": safe_float(precision_score(y_test, predicted, zero_division=0)),
        "recall": safe_float(recall_score(y_test, predicted, zero_division=0)),
        "f1_score": safe_float(f1_score(y_test, predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions,
    }


def adaptive_analysis(
    mappings: dict[str, str | None],
    predictive: dict[str, Any],
    individual: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_test = {row["Mapped column"]: row for row in individual}
    rows: list[dict[str, Any]] = []
    fail_rates = []
    for key, title in CANONICAL_TESTS.items():
        column = mappings.get(key)
        if column and column in by_test:
            rate = safe_float(by_test[column]["Fail percentage"]) / 100
        else:
            rate = 0.0
        fail_rates.append((title, rate, bool(key == "supply_voltage")))

    max_rate = max((rate for _, rate, _ in fail_rates), default=0.0)
    for title, rate, critical in fail_rates:
        if critical:
            priority = "MANDATORY"
        elif rate > 0:
            normalized_priority = rate / max_rate if max_rate else 0
            priority = "HIGH" if normalized_priority >= 0.7 else "MEDIUM" if normalized_priority >= 0.3 else "LOW"
        else:
            priority = "LOW"
        rows.append({
            "Recommended test": title,
            "Priority": priority,
            "Observed failure rate": round(rate * 100, 3),
            "Reason": "Safety/supply-related test retained in mandatory full flow." if critical
            else "Priority estimated from observed failure rate; no learned score was used to weaken mandatory coverage.",
        })
    return sorted(rows, key=lambda row: {"MANDATORY": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[row["Priority"]])


def analyze_uploaded(
    frame: pd.DataFrame,
    mappings: dict[str, str | None],
    label_column: str | None = None,
) -> dict[str, Any]:
    cleaned, cleaning = clean_dataset(frame)
    label, inferred_label_column = infer_label(cleaned, label_column)
    individual, missing = individual_test_analysis(cleaned, mappings, label)
    static = static_reduction(cleaned, mappings)
    predictive = predictive_analysis(cleaned, label, inferred_label_column)
    adaptive = adaptive_analysis(mappings, predictive, individual)

    time_column = mappings.get("measurement_time")
    if time_column and time_column in cleaned.columns:
        original_time = float(pd.to_numeric(cleaned[time_column], errors="coerce").sum(skipna=True))
        measurement_unit = "ms across records" if "ms" in normalized(time_column) else "recorded units across records"
        removable = [row for row in static if row["Decision"] == "Potential review candidate"]
        estimated_savings = original_time * (len(removable) / max(len(static), 1)) if original_time else 0.0
        optimized_time = max(0.0, original_time - estimated_savings)
        time_estimation = "Estimated by proportion of zero-variance, no-failure candidate tests; validate before production use."
    else:
        original_time = 0.0
        optimized_time = 0.0
        estimated_savings = 0.0
        measurement_unit = "Unavailable (measurement-time column not mapped)"
        time_estimation = "Actual measurement time unavailable; no percentage improvement claimed."

    return {
        "cleaning": cleaning,
        "missing_tests": missing,
        "individual_tests": individual,
        "static_reduction": static,
        "measurement_efficiency": {
            "measurement_time_column": time_column,
            "measurement_unit": measurement_unit,
            "original": original_time,
            "optimized_estimated": optimized_time,
            "time_saved": estimated_savings,
            "improvement_pct": round(estimated_savings / original_time * 100, 3) if original_time else None,
            "basis": time_estimation,
            "type": "estimated" if original_time and estimated_savings else "not available",
        },
        "predictive": {
            key: value for key, value in predictive.items() if key != "predictions"
        },
        "predictive_predictions": predictive["predictions"].fillna(0).to_dict(orient="records") if predictive.get("available") else [],
        "adaptive": adaptive,
        "processed_dataset": cleaned,
    }
