# Run the complete FBG analysis

The pipeline regenerates Phases 3–9 from raw recordings. It never reads archived
`results/` or `data/processed/` files and never overwrites an existing run directory.

## Install

Use Python 3.11 or newer (CI uses 3.11). From the repository directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_pipeline.py
```

On Linux/macOS use `.venv/bin/python` instead. A GPU is not required.

## Commands

```powershell
python run_pipeline.py --output runs/my-analysis
python run_pipeline.py --core-only --output runs/core-analysis
python run_pipeline.py --input raw_data --output runs/another-analysis
python -m unittest discover -s tests -v
```

The default includes PCA, UMAP, clustering, autoencoder and explainability.
`--core-only` ends after Phase 8. Output paths must not already exist. The default
output is a uniquely named directory under `runs/`. Initial UMAP compilation can
take longer than later runs. Progress appears per stage; full details are in logs.

## What happens

1. Check dependencies, filenames, eight-column input, finite measurements,
   minimum recording length and uniformly increasing 50 Hz timestamps.
2. Copy the inputs into a new workspace and record SHA-256 hashes.
3. Execute each existing phase in a separate process, using a noninteractive plot
   backend. Phase 7 precedes Phase 8 because Phase 8 consumes its statistics.
4. Check exit codes, validation failures, required outputs, case counts,
   duplicate keys, finite spectral features and bounded engineering indices.
5. Generate `REPORT.md`, `manifest.json`, `environment.txt`, stage logs, processed
   signals, all stage result tables, plots and trained autoencoder weights.

Failures return a nonzero exit code, stop downstream execution, and leave logs
and the manifest for diagnosis. Choose a new directory to retry after fixing the
cause. No automatic deletion or reuse of previous results occurs.

## Input and scientific scope

The existing analysis is designed for the RVCE Expert recordings. Names must
contain `expert<number>.`, with a unique number per recording. Measurements must
have the same three channels and 50 Hz acquisition configuration. Several legacy
phase reports and validation checks explicitly describe the seven-recording,
12-impact benchmark; arbitrary new datasets are not yet supported by all ML
stages. The pipeline fails rather than silently accepting incompatible results.

Results are offline exploratory analysis, not a deployed prediction service.
Detector agreement is not independently measured accuracy. Training reconstruction
and classification performance are not held-out evaluation. Missing event features
remain missing rather than being treated as zero. Model weights are research
artifacts; an inference service would also require a saved preprocessing contract.

The generated run report is the summary for that execution. Archived results and
older README scientific claims are historical and must not override fresh tables.

## Continuous integration

GitHub Actions installs dependencies, runs unit tests and regenerates the full
analysis on Linux. It uploads the run folder even on failure. CI does not push
generated results into the repository. Each local run saves its package versions;
library ranges in `requirements.txt` are compatibility constraints, not a lockfile.
