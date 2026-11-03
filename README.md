# DC Parametric Test-Time Optimization

This project implements the methodology in `data/AI-Optimized DC Parametric Test Time Reduction - Methodology.pdf`
using the synthetic data in `data/`.

## Setup

```bash
python -m pip install -r requirements.txt
python -m dcopt.io --config config.yaml
python -m pytest
```

## Web interface

Start the FastAPI backend in one terminal:

```bash
uvicorn api:app --reload --port 8000
```

Start the Streamlit frontend in another terminal:

```bash
streamlit run app.py
```

The Streamlit UI defaults to `http://localhost:8000`. Upload and analyze a CSV
or Excel dataset from **Upload & Analyze**.

For Streamlit Community Cloud, no local FastAPI process is available. Set
`DCOPT_API_URL` only when a public FastAPI deployment is also used; otherwise
uploaded-file analysis runs directly in Streamlit.

## Rules

- Split data by lot; never by device.
- `is_latent_defect_truth` is used only to score PAT.
- Models only return RUN or SKIP decisions.
- Fail decisions come from measured values versus catalog limits.
- Errors or uncertainty fail safe to the full flow.
- Configuration values are authoritative; do not hard-code decision thresholds.

## Reproducible foundation

```bash
make phase0
make io
make test
```
