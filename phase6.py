import os
import sys
import numpy as np
import pandas as pd

from src.phase_6.fft_analysis import (
    extract_fft_features
)

from src.phase_6.wavelet_analysis import (
    extract_wavelet_features
)

# Phase 4 detection methods (read-only usage to get impact times)
from peak_detection import detect_peak
from threshold_detection import detect_threshold
from derivative_detection import detect_derivative
from changepoint_detection import detect_changepoint


INPUT_DIR = r"data\processed\final_phase_input"

OUTPUT_DIR = r"results\phase6"

PHASE4_RESULTS = r"results\phase4\phase4_method_results.csv"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD PHASE 4 DETECTION RESULTS
# ============================================================

phase4_df = pd.read_csv(PHASE4_RESULTS)

# Build lookup: (Dataset, FBG) -> Final_Result
# e.g. ("Expert 7", "FBG1") -> "NO IMPACT"
phase4_lookup = {}

for _, row in phase4_df.iterrows():
    key = (
        row["Dataset"].strip(),
        row["FBG"].strip()
    )
    phase4_lookup[key] = row["Final_Result"].strip()


# ============================================================
# MAP FILENAME TO EXPERT NAME
# ============================================================

def extract_expert_name(filename):
    """Extract 'Expert N' from filename like
    'RVCE mechanical strain expert7...csv'
    """
    base = os.path.basename(filename)
    try:
        num = int(
            base.split("expert")[1].split(".")[0]
        )
        return f"Expert {num}"
    except (IndexError, ValueError):
        return None


# ============================================================
# GET IMPACT TIME FROM PHASE 4 DETECTION METHODS
# ============================================================

def get_phase4_impact_time(time_series, signal_series):
    """
    Re-run the same 4 Phase 4 detection methods to obtain
    the impact time. Returns the earliest detection time
    among methods that detected an impact (consensus time).

    This does NOT re-decide impact/no-impact — that decision
    comes from Phase 4 results CSV. This only recovers the
    impact time for windowing.
    """
    peak_res, peak_time, *_ = detect_peak(
        time_series, signal_series
    )

    thresh_res, thresh_time, *_ = detect_threshold(
        time_series, signal_series
    )

    deriv_res, deriv_time, *_ = detect_derivative(
        time_series, signal_series
    )

    cp_res, cp_time, *_ = detect_changepoint(
        time_series, signal_series
    )

    # Collect times from methods that detected impact
    detected_times = []

    if peak_res and peak_time is not None:
        detected_times.append(peak_time)

    if thresh_res and thresh_time is not None:
        detected_times.append(thresh_time)

    if deriv_res and deriv_time is not None:
        detected_times.append(deriv_time)

    if cp_res and cp_time is not None:
        detected_times.append(cp_time)

    if detected_times:
        # Use earliest detection as impact onset
        return min(detected_times)

    return None


# ============================================================
# EXTRACT IMPACT WINDOW
# ============================================================

def extract_impact_window(time, signal, impact_time,
                          sampling_interval):
    """
    Extract a symmetric window around the Phase 4 detected
    impact time. Window extends 50 samples before and 50
    samples after the impact index (or to signal boundaries).
    """
    # Find the index closest to impact_time
    impact_idx = np.argmin(
        np.abs(time - impact_time)
    )

    # Window: 50 samples before, 50 samples after
    half_window = 50

    start_idx = max(0, impact_idx - half_window)
    end_idx = min(
        len(signal),
        impact_idx + half_window + 1
    )

    return signal[start_idx:end_idx].copy(), sampling_interval


# ============================================================
# NaN FEATURE DICTIONARIES
# ============================================================

NAN_FFT_FEATURES = {
    "Dominant_Frequency": np.nan,
    "Spectral_Energy": np.nan,
    "Spectral_Entropy": np.nan,
    "Spectral_Centroid": np.nan,
    "Bandwidth": np.nan
}

NAN_WAVELET_FEATURES = {
    "Approximation_Energy": np.nan,
    "Detail_Energy": np.nan,
    "Wavelet_Energy": np.nan,
    "Wavelet_Entropy": np.nan,
    "Detail_Approx_Ratio": np.nan
}


# ============================================================
# MAIN PROCESSING
# ============================================================

all_results = []

print("\nPHASE 6 STARTED\n")

total_impact = 0
total_no_impact = 0

for filename in os.listdir(INPUT_DIR):

    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(
        INPUT_DIR,
        filename
    )

    expert_name = extract_expert_name(filename)

    if expert_name is None:
        print(f"WARNING: Cannot parse expert from {filename}")
        continue

    print(f"Processing: {filename}")

    df = pd.read_csv(file_path)

    time = df["Time"].values

    sampling_interval = np.mean(
        np.diff(time)
    )

    sensors = [
        "FBG1_processed",
        "FBG2_processed",
        "FBG3_processed"
    ]

    # Map sensor column name to FBG name
    sensor_to_fbg = {
        "FBG1_processed": "FBG1",
        "FBG2_processed": "FBG2",
        "FBG3_processed": "FBG3"
    }

    for sensor in sensors:

        fbg = sensor_to_fbg[sensor]

        # Look up Phase 4 result
        lookup_key = (expert_name, fbg)
        phase4_result = phase4_lookup.get(
            lookup_key, None
        )

        if phase4_result is None:
            print(
                f"  WARNING: No Phase 4 result for "
                f"{expert_name} {fbg}, skipping"
            )
            continue

        signal = df[sensor].values

        if phase4_result == "IMPACT":

            total_impact += 1

            # Get impact time from Phase 4 methods
            impact_time = get_phase4_impact_time(
                df["Time"],
                df[sensor]
            )

            if impact_time is None:
                # Phase 4 said IMPACT but no time recovered
                # (should not happen, but handle gracefully)
                print(
                    f"  WARNING: {expert_name} {fbg} - "
                    f"IMPACT but no time found"
                )
                fft_features = NAN_FFT_FEATURES.copy()
                wavelet_features = NAN_WAVELET_FEATURES.copy()
            else:
                # Extract window around impact
                windowed_signal, _ = extract_impact_window(
                    time, signal,
                    impact_time, sampling_interval
                )

                print(
                    f"  {expert_name} {fbg}: IMPACT at "
                    f"t={impact_time:.4f}s, "
                    f"window={len(windowed_signal)} samples"
                )

                fft_features = extract_fft_features(
                    windowed_signal,
                    sampling_interval
                )

                wavelet_features = extract_wavelet_features(
                    windowed_signal
                )

        else:
            # NO IMPACT — features are NaN
            total_no_impact += 1

            print(
                f"  {expert_name} {fbg}: NO IMPACT — "
                f"features set to NaN"
            )

            fft_features = NAN_FFT_FEATURES.copy()
            wavelet_features = NAN_WAVELET_FEATURES.copy()

        row = {
            "File": filename,
            "Sensor": sensor,
            "Phase4_Result": phase4_result
        }

        row.update(
            fft_features
        )

        row.update(
            wavelet_features
        )

        all_results.append(
            row
        )


results_df = pd.DataFrame(
    all_results
)

output_file = os.path.join(
    OUTPUT_DIR,
    "phase6_multidomain_features.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nPHASE 6 COMPLETE")

print(
    f"\nResults saved to:\n{output_file}"
)

print(f"\nTotal recordings: {len(all_results)}")
print(f"IMPACT cases: {total_impact}")
print(f"NO IMPACT cases: {total_no_impact}")

print("\nFirst 5 Results:\n")

print(
    results_df.head()
)


# ============================================================
# PHASE 6 VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("PHASE 6 VALIDATION")
print("=" * 60)

validation_passed = True

# --- A. Input structure: 7 datasets x 3 FBG = 21 recordings ---

expected_recordings = 7 * 3  # 21

actual_recordings = len(results_df)

if actual_recordings == expected_recordings:
    print(
        f"[PASS] A. Input structure: "
        f"{actual_recordings} recordings "
        f"(7 datasets x 3 FBG)"
    )
else:
    print(
        f"[FAIL] A. Input structure: "
        f"Expected {expected_recordings}, "
        f"got {actual_recordings}"
    )
    validation_passed = False

# --- B. 10 features per valid IMPACT recording ---

expected_feature_cols = [
    "Dominant_Frequency",
    "Spectral_Energy",
    "Spectral_Entropy",
    "Spectral_Centroid",
    "Bandwidth",
    "Approximation_Energy",
    "Detail_Energy",
    "Wavelet_Energy",
    "Wavelet_Entropy",
    "Detail_Approx_Ratio"
]

# Check all feature columns exist
missing_cols = [
    c for c in expected_feature_cols
    if c not in results_df.columns
]

if not missing_cols:
    print(
        f"[PASS] B. All 10 feature columns present"
    )
else:
    print(
        f"[FAIL] B. Missing feature columns: "
        f"{missing_cols}"
    )
    validation_passed = False

# Check IMPACT rows have all 10 features non-NaN
impact_rows = results_df[
    results_df["Phase4_Result"] == "IMPACT"
]

impact_features_valid = True

for idx, row in impact_rows.iterrows():
    nan_features = [
        c for c in expected_feature_cols
        if pd.isna(row[c])
    ]
    if nan_features:
        print(
            f"[FAIL] B. IMPACT row {row['File']} "
            f"{row['Sensor']} has NaN in: "
            f"{nan_features}"
        )
        impact_features_valid = False
        validation_passed = False

if impact_features_valid:
    print(
        f"[PASS] B. All {len(impact_rows)} IMPACT "
        f"recordings have 10 valid features"
    )

# --- C. NO IMPACT features must be NaN ---

no_impact_rows = results_df[
    results_df["Phase4_Result"] == "NO IMPACT"
]

no_impact_valid = True

for idx, row in no_impact_rows.iterrows():
    non_nan = [
        c for c in expected_feature_cols
        if not pd.isna(row[c])
    ]
    if non_nan:
        print(
            f"[FAIL] C. NO IMPACT row {row['File']} "
            f"{row['Sensor']} has non-NaN features: "
            f"{non_nan}"
        )
        no_impact_valid = False
        validation_passed = False

if no_impact_valid:
    print(
        f"[PASS] C. All {len(no_impact_rows)} NO IMPACT "
        f"recordings have NaN features"
    )

# --- D. Frequency sanity (Nyquist check) ---

fs = 1.0 / sampling_interval  # last file's fs
nyquist = fs / 2.0

freq_valid = True

for idx, row in impact_rows.iterrows():
    dom_freq = row["Dominant_Frequency"]
    centroid = row["Spectral_Centroid"]

    if not pd.isna(dom_freq) and dom_freq > nyquist:
        print(
            f"[FAIL] D. {row['File']} {row['Sensor']} "
            f"Dominant_Frequency={dom_freq:.2f} > "
            f"Nyquist={nyquist:.2f}"
        )
        freq_valid = False
        validation_passed = False

    if not pd.isna(centroid) and centroid > nyquist:
        print(
            f"[FAIL] D. {row['File']} {row['Sensor']} "
            f"Spectral_Centroid={centroid:.2f} > "
            f"Nyquist={nyquist:.2f}"
        )
        freq_valid = False
        validation_passed = False

if freq_valid:
    print(
        f"[PASS] D. All FFT frequencies within "
        f"Nyquist range (fs={fs:.2f} Hz, "
        f"Nyquist={nyquist:.2f} Hz)"
    )

# --- E. No unexpected missing/invalid values ---

e_valid = True

for idx, row in impact_rows.iterrows():
    for c in expected_feature_cols:
        val = row[c]
        if pd.isna(val):
            print(
                f"[FAIL] E. Unexpected NaN in IMPACT "
                f"row: {row['File']} {row['Sensor']} "
                f"feature={c}"
            )
            e_valid = False
            validation_passed = False
        elif np.isinf(val):
            print(
                f"[FAIL] E. Inf value in IMPACT "
                f"row: {row['File']} {row['Sensor']} "
                f"feature={c}"
            )
            e_valid = False
            validation_passed = False

if e_valid:
    print(
        f"[PASS] E. No unexpected missing/invalid "
        f"values in IMPACT recordings"
    )

# --- Final validation result ---

print("\n" + "-" * 40)

if validation_passed:
    print("PHASE 6 VALIDATION: ALL CHECKS PASSED")
else:
    print("PHASE 6 VALIDATION: SOME CHECKS FAILED")

print("=" * 60)