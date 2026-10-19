# FBG Impact Analysis

A Python research workflow for comparing impact responses from fiber Bragg
grating sensors with copper, bare-fiber and steel configurations.

An FBG sensor reflects a wavelength of light that changes as it deforms. This
project analyzes those wavelength changes, identifies impact-like responses,
measures their characteristics and explores differences between sensor configurations.

## Run the complete pipeline

Use Python 3.11 or newer. CI uses Python 3.11 on Linux; Windows is also supported.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_pipeline.py
```

On Linux/macOS use `.venv/bin/python` instead of `.venv\Scripts\python`.

The command executes Phases 3–9 from raw inputs in a new folder under `runs/`.
It does not reuse or overwrite archived result files. Open the generated
`REPORT.md` for the summary. Installation downloads the scientific and ML
dependencies; a GPU is not required.

For named outputs or analysis without the optional ML stages:

```powershell
python run_pipeline.py --output runs/demo
python run_pipeline.py --core-only --output runs/core-demo
python -m unittest discover -s tests -v
```

Choose a new output directory on every run. See [PIPELINE.md](PIPELINE.md) for
input requirements, failure handling, reproducibility and current limitations.

## What each stage does

| Stage | Script | Purpose |
|---|---|---|
| 3 | `phase3_main.py` | Read recordings, clean, subtract baseline, compare filters and export Savitzky–Golay signals |
| 4 | `phase4.py` | Run peak, threshold, derivative and change-point detectors; at least two positive decisions imply IMPACT |
| 5 | `phase5.py` | Extract time-domain response measurements and compare materials |
| 6 | `phase6.py` | Calculate FFT and wavelet features around detected times |
| 7 | `phase7.py` | Summarize counts, means, variation and confidence intervals |
| 8 | `phase8.py` | Combine normalized measurements into DSTI, PEI, SII and DRI scores |
| 9 | `phase9_pca.py`, `phase9_umap.py` | Explore patterns through dimensionality reduction |
| 9 part 2 | `phase9_part2.py` | Coordinate clustering, autoencoder reconstruction and exploratory feature attribution |

Phase 8 reads Phase 7 outputs, so the runner executes Phase 7 first. Each stage
runs in a separate process and has its own log. Validation failures stop the run.

## Project structure

```text
raw_data/                Original eight-column experimental recordings
src/io/                  Loading and saving measurements
src/preprocessing/       Cleaning and baseline subtraction
src/filtering/           Four reusable filters
src/analysis/            Filter/noise evaluation
src/visualization/       Signal plotting functions
src/engineering_features.py  Individual time-domain calculations
src/feature_extraction.py    Event windowing and feature collection
src/comparison_analysis.py   Material tables, plots and reports
src/phase_6/             FFT and wavelet calculations
*_detection.py           Four impact-detection methods
run_pipeline.py          Full workflow, input checks, logs, manifest and report
tests/                   Input, artifact and mathematical-invariant tests
runs/                    New run outputs (ignored by Git)
results/                 Historical committed analysis outputs
data/processed/          Historical committed processed signals
```

Each run contains copied inputs, processed signals, stage outputs, logs, a summary,
package versions and a manifest with source/input/output hashes and stage status.
GitHub Actions runs the tests and full pipeline and uploads its run artifacts.

## Dataset and interpretation

The repository contains Experts 7–13: seven recordings with three channels each.
The mapping is FBG1 = Copper, FBG2 = Bare and FBG3 = Steel. Sampling is configured
at 50 Hz. Phase 3 subtracts an initial 100-sample baseline; detectors separately
estimate a quiet early baseline. The final exported filter is Savitzky–Golay.

Historical Phase 4 results classify 12 of 21 recording-channel cases as IMPACT:
Bare 7, Copper 3, Steel 2. These are algorithm classifications, not independently
verified physical-event counts or accuracy measurements. Fresh run tables are
the authority for that execution.

The historical first two PCA components retain about 55.8% of variation; this is
not classification accuracy. Clustering shows partial structure with overlap.
The autoencoder reconstructs its training data; the Random Forest is supervised
exploratory analysis. Its training accuracy does not establish performance on
unseen experiments. Attribution uses a custom approximate Shapley-style method,
not the standard SHAP package.

Engineering indices are dataset-normalized composite scores, not certified
material ratings. Wavelet coefficient energy is not impact energy in joules.
Higher detail energy or entropy alone does not establish better transient sensing.
Matched windows, baseline-noise comparisons and independent impact labels are
needed for stronger comparisons.

The complete workflow is an offline research pipeline. It is not a live sensor
service or a validated predictive system. Existing stage reports contain some
benchmark-specific narratives and checks; extending to arbitrary datasets
requires revising those assumptions and evaluating new data independently.
