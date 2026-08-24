# FBG Signal Preprocessing and Filtering

## Overview

This repository contains the first three phases of an FBG (Fiber Bragg Grating) signal processing pipeline developed for impact sensing applications.

The objective of this work is to transform raw FBG wavelength data into clean, noise-reduced signals suitable for further analysis. The repository focuses on data loading, preprocessing, noise reduction, and filter performance evaluation.

---

## Project Workflow

```text
Raw FBG Data
      ↓
Data Loading
      ↓
Signal Cleaning
      ↓
Baseline Correction
      ↓
Wavelength Shift Calculation
      ↓
Signal Filtering
      ↓
Noise Analysis
      ↓
Filter Comparison
```

---

## Phase 1: Data Loading

This phase handles the ingestion of raw FBG sensor data.

### Tasks

* Load raw sensor files
* Validate data integrity
* Organize data for processing
* Store processed outputs

### Modules

* `src/io/data_loader.py`
* `src/io/save_processed_data.py`

---

## Phase 2: Signal Preprocessing

This phase prepares the raw signal for analysis by removing inconsistencies and correcting baseline variations.

### Tasks

* Data cleaning
* Missing value handling
* Baseline correction
* Wavelength shift computation
* Preprocessing pipeline execution

### Modules

* `src/preprocessing/cleaning.py`
* `src/preprocessing/baseline.py`
* `src/preprocessing/wavelength.py`
* `src/preprocessing/preprocessing_pipeline.py`

---

## Phase 3: Signal Filtering and Noise Reduction

This phase focuses on reducing measurement noise and improving signal quality.

### Implemented Filters

#### Moving Average Filter

Provides simple smoothing by averaging neighboring samples.

#### Median Filter

Reduces impulsive noise while preserving important signal features.

#### Butterworth Filter

Performs frequency-domain smoothing using a low-pass filter.

#### Savitzky-Golay Filter

Preserves signal shape and peak characteristics while reducing noise. This filter was selected as the preferred filtering approach for subsequent work.

### Additional Analysis

* Noise analysis
* Filter performance comparison
* Visualization of filtered signals

### Modules

* `src/filtering/moving_average.py`
* `src/filtering/median.py`
* `src/filtering/butterworth.py`
* `src/filtering/savitzky_golay.py`
* `src/filtering/filter_comparison.py`
* `src/analysis/noise_analysis.py`
* `src/visualization/plots.py`

---

## Repository Structure

```text
FBG_Phase3/
│
├── raw_data/
│
├── results/
│
├── src/
│   ├── io/
│   ├── preprocessing/
│   ├── filtering/
│   ├── analysis/
│   ├── visualization/
│   └── config.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python main.py
```

---

## Current Status

This repository contains development up to **Phase 3: Signal Filtering and Noise Reduction**.

Subsequent phases involving impact detection, feature extraction, ensemble methods, and advanced evaluation are intentionally excluded from this repository.
