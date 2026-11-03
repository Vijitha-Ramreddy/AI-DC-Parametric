from __future__ import annotations

import importlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dcopt.upload_analysis import (
    CANONICAL_TESTS,
    analyze_uploaded,
    clean_dataset,
    read_uploaded,
    suggest_mappings,
)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
ALLOWED_MODULES = {
    "io": "dcopt.io",
    "phase0": "dcopt.phase0_data_foundation",
    "baseline": "dcopt.baseline",
    "settling": "dcopt.settling",
    "ordering": "dcopt.ordering",
    "static": "dcopt.static_reduction",
    "predictive": "dcopt.predictive",
    "adaptive": "dcopt.adaptive",
    "pat": "dcopt.pat",
    "simulator": "dcopt.simulator",
    "monitoring": "dcopt.monitoring",
}
DEFAULT_PIPELINE = [
    "io", "phase0", "baseline", "settling", "ordering", "static",
    "predictive", "adaptive", "pat", "simulator", "monitoring",
]

app = FastAPI(title="DC Parametric Test-Time Optimization API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRequest(BaseModel):
    modules: list[Literal[
        "io", "phase0", "baseline", "settling", "ordering", "static",
        "predictive", "adaptive", "pat", "simulator", "monitoring",
    ]] | None = None


class UploadAnalysisRequest(BaseModel):
    upload_id: str
    mappings: dict[str, str | None]
    label_column: str | None = None


def results_root() -> Path:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    root = Path(config["paths"]["results_dir"])
    return root if root.is_absolute() else ROOT / root


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported.")

    upload_dir = ROOT / "results" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    target = upload_dir / f"{upload_id}{suffix}"
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        frame = read_uploaded(target)
        cleaned, cleaning = clean_dataset(frame)
        suggestions = suggest_mappings(frame)
        preview = cleaned.head(20).fillna("").to_dict(orient="records")
        return {
            "upload_id": upload_id,
            "filename": file.filename,
            "columns": cleaned.columns.tolist(),
            "preview": preview,
            "cleaning": cleaning,
            "suggested_mappings": suggestions,
            "missing_required_tests": [
                title for key, title in CANONICAL_TESTS.items()
                if not suggestions.get(key)
            ],
        }
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Unable to read dataset: {exc}") from exc


@app.post("/api/upload/analyze")
def analyze_upload(request: UploadAnalysisRequest) -> dict:
    upload_dir = ROOT / "results" / "uploads"
    matches = list(upload_dir.glob(f"{request.upload_id}.*"))
    if len(matches) != 1:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found.")
    try:
        frame = read_uploaded(matches[0])
        result = analyze_uploaded(frame, request.mappings, request.label_column)
        processed = result.pop("processed_dataset")
        processed_path = upload_dir / f"{request.upload_id}_processed.csv"
        processed.to_csv(processed_path, index=False)
        result["processed_download"] = f"/api/upload/{request.upload_id}/processed.csv"
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}") from exc


@app.get("/api/upload/{upload_id}/processed.csv")
def download_processed(upload_id: str) -> FileResponse:
    path = ROOT / "results" / "uploads" / f"{upload_id}_processed.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Processed dataset not found.")
    return FileResponse(path, media_type="text/csv", filename="processed_dataset.csv")


@app.get("/api/data/quality")
def data_quality() -> dict:
    module = importlib.import_module("dcopt.io")
    datasets = module.load_datasets(CONFIG_PATH)
    return module.data_quality_report(datasets)


@app.get("/api/results")
def list_results() -> dict[str, list[str]]:
    root = results_root()
    if not root.exists():
        return {"stages": []}
    stages = sorted(
        item.name for item in root.iterdir()
        if item.is_dir() and any(child.is_file() for child in item.iterdir())
    )
    return {"stages": stages}


@app.get("/api/results/{stage}")
def stage_artifacts(stage: str) -> dict[str, list[str]]:
    root = results_root()
    directory = (root / stage).resolve()
    if not directory.is_relative_to(root.resolve()) or not directory.is_dir():
        raise HTTPException(status_code=404, detail="Result stage not found")
    files = sorted(
        item.name for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in {".csv", ".md", ".png"}
    )
    return {"files": files}


@app.get("/api/results/{stage}/{filename}")
def read_artifact(
    stage: str,
    filename: str,
    limit: int = Query(default=1000, ge=1, le=10000),
):
    directory = (results_root() / stage).resolve()
    artifact = (directory / filename).resolve()
    if not artifact.is_relative_to(results_root().resolve()) or not artifact.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    suffix = artifact.suffix.lower()
    if suffix == ".png":
        return FileResponse(artifact, media_type="image/png")
    if suffix == ".csv":
        frame = pd.read_csv(artifact)
        return {
            "columns": frame.columns.tolist(),
            "rows": frame.head(limit).fillna("").to_dict(orient="records"),
            "total_rows": len(frame),
        }
    if suffix == ".md":
        return {"content": artifact.read_text(encoding="utf-8")}
    raise HTTPException(status_code=404, detail="Unsupported artifact")


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRequest) -> dict:
    modules = request.modules or DEFAULT_PIPELINE
    completed: list[str] = []
    for module_name in modules:
        try:
            module = importlib.import_module(ALLOWED_MODULES[module_name])
            output = module.run(CONFIG_PATH)
            completed.append(module_name)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"stage": module_name, "message": str(exc), "completed": completed},
            ) from exc
    return {"status": "ok", "completed": completed, "results_dir": str(results_root())}
