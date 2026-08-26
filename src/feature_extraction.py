import os
import glob
import pandas as pd
import numpy as np

from src.engineering_features import (
    calculate_baseline,
    calculate_peak_shift,
    calculate_residual_shift,
    calculate_rise_time,
    calculate_recovery_time,
    calculate_peak_width,
    calculate_max_slope,
    calculate_rms,
    calculate_signal_energy,
    calculate_peak_to_peak,
    calculate_variance,
    calculate_std,
    calculate_entropy,
    calculate_auc
)

# Material mapping for FBG channels
MATERIAL_MAP = {
    "FBG1": "Bare",
    "FBG2": "Copper",
    "FBG3": "Steel"
}

# Unit mapping for all 13 features
FEATURE_UNITS = {
    "peak_shift_signed": "nm",
    "peak_shift_abs": "nm",
    "peak_time": "s",
    "residual_shift_signed": "nm",
    "residual_shift_abs": "nm",
    "recovered_level": "nm",
    "rise_time_seconds": "s",
    "recovery_time_seconds": "s",
    "recovery_timestamp": "s",
    "peak_width_seconds": "s",
    "max_slope_pos": "nm/s",
    "max_slope_neg": "nm/s",
    "max_slope_abs": "nm/s",
    "rms": "nm",
    "signal_energy": "nm²·s",
    "peak_to_peak": "nm",
    "variance": "nm²",
    "std_dev": "nm",
    "entropy": "bits",
    "auc_signed": "nm·s",
    "auc_abs": "nm·s"
}


def extract_features_for_signal(time_series, signal_series, impact_detected=True, baseline_samples=100):
    """
    Extract all 13 engineering features for a single FBG time-series signal.
    
    Parameters:
        time_series (pd.Series or np.ndarray): Time vector in seconds.
        signal_series (pd.Series or np.ndarray): Processed wavelength shift in nm.
        impact_detected (bool): Whether an impact event was detected in Phase 4.
        baseline_samples (int): Number of initial samples for pre-impact baseline.
        
    Returns:
        dict: Complete feature dictionary with key-value pairs.
    """
    time = np.asarray(time_series)
    signal = np.asarray(signal_series)
    
    # 1. Baseline calculation
    baseline_val, noise_std = calculate_baseline(signal, baseline_samples=baseline_samples)
    signal_corrected = signal - baseline_val
    
    # 2. Peak Shift & Timing
    peak_shift_signed, peak_shift_abs, peak_time, peak_idx = calculate_peak_shift(signal_corrected, time)
    
    # 3. Recovery Time & Event Window Boundaries
    recovery_time_seconds, recovery_timestamp, recovery_end_idx = calculate_recovery_time(
        signal_corrected, time, peak_idx, noise_std
    )
    
    # 4. Residual Shift
    residual_shift_signed, residual_shift_abs, recovered_level = calculate_residual_shift(
        signal_corrected, time, recovery_end_idx=recovery_end_idx
    )
    
    # 5. Impact Event Window Selection
    # For baseline-corrected feature analysis, analyze event region around impact
    impact_start_idx = max(0, peak_idx - 50)
    event_signal = signal_corrected[impact_start_idx:min(len(signal_corrected), recovery_end_idx + 10)]
    event_time = time[impact_start_idx:min(len(time), recovery_end_idx + 10)]
    
    if len(event_signal) < 5:
        event_signal = signal_corrected
        event_time = time
        
    rel_peak_idx = peak_idx - impact_start_idx
    if rel_peak_idx < 0 or rel_peak_idx >= len(event_signal):
        rel_peak_idx = int(np.argmax(np.abs(event_signal)))
        
    # 6. Rise Time
    rise_time_seconds = calculate_rise_time(event_signal, event_time, rel_peak_idx, impact_start_idx=0)
    
    # 7. Peak Width (FWHM)
    peak_width_seconds = calculate_peak_width(event_signal, event_time, rel_peak_idx)
    
    # 8. Maximum Slope
    max_slope_pos, max_slope_neg, max_slope_abs = calculate_max_slope(event_signal, event_time)
    
    # 9. RMS
    rms_val = calculate_rms(event_signal)
    
    # 10. Signal Energy
    signal_energy = calculate_signal_energy(event_signal, event_time)
    
    # 11. Peak-to-Peak
    peak_to_peak = calculate_peak_to_peak(event_signal)
    
    # 12. Variance
    variance_val = calculate_variance(event_signal)
    
    # 13. Standard Deviation
    std_val = calculate_std(event_signal)
    
    # 14. Entropy
    entropy_bits = calculate_entropy(event_signal)
    
    # 15. Area Under Curve (AUC)
    auc_signed, auc_abs = calculate_auc(event_signal, event_time)
    
    features = {
        "baseline_nm": baseline_val,
        "noise_std_nm": noise_std,
        "peak_shift_signed": peak_shift_signed,
        "peak_shift_abs": peak_shift_abs,
        "peak_time": peak_time,
        "residual_shift_signed": residual_shift_signed,
        "residual_shift_abs": residual_shift_abs,
        "recovered_level": recovered_level,
        "rise_time_seconds": rise_time_seconds,
        "recovery_time_seconds": recovery_time_seconds,
        "recovery_timestamp": recovery_timestamp,
        "peak_width_seconds": peak_width_seconds,
        "max_slope_pos": max_slope_pos,
        "max_slope_neg": max_slope_neg,
        "max_slope_abs": max_slope_abs,
        "rms": rms_val,
        "signal_energy": signal_energy,
        "peak_to_peak": peak_to_peak,
        "variance": variance_val,
        "std_dev": std_val,
        "entropy": entropy_bits,
        "auc_signed": auc_signed,
        "auc_abs": auc_abs
    }
    
    return features


def extract_all_dataset_features(data_dir="data/processed/final_phase_input", phase4_dir="results/phase4"):
    """
    Extract Phase 5 features for all datasets and sensors in the repository.
    
    Returns:
        tuple: (all_features_df, long_format_df)
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    def extract_expert_num(filename):
        base = os.path.basename(filename)
        try:
            return int(base.split("expert")[1].split(".")[0])
        except (IndexError, ValueError):
            return 0
            
    csv_files.sort(key=extract_expert_num)
    
    # Load Phase 4 method results if present
    phase4_results_path = os.path.join(phase4_dir, "phase4_method_results.csv")
    phase4_df = None
    if os.path.exists(phase4_results_path):
        phase4_df = pd.read_csv(phase4_results_path)
        
    records = []
    long_records = []
    
    fbgs = ["FBG1", "FBG2", "FBG3"]
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        expert_num = extract_expert_num(filename)
        expert_name = f"Expert {expert_num}"
        
        df = pd.read_csv(file_path)
        if "Time" not in df.columns:
            continue
            
        time_series = df["Time"]
        
        for fbg in fbgs:
            col_name = f"{fbg}_processed"
            if col_name not in df.columns:
                continue
                
            signal_series = df[col_name]
            material = MATERIAL_MAP.get(fbg, "Unknown")
            
            # Check Phase 4 detection status
            impact_status = "UNKNOWN"
            methods_detected = 0
            if phase4_df is not None:
                match = phase4_df[(phase4_df["Dataset"] == expert_name) & (phase4_df["FBG"] == fbg)]
                if not match.empty:
                    impact_status = match["Final_Result"].values[0]
                    methods_detected = int(match["Methods_Detected"].values[0])
                    
            feats = extract_features_for_signal(
                time_series,
                signal_series,
                impact_detected=(impact_status == "IMPACT")
            )
            
            row = {
                "Dataset": expert_name,
                "Expert_Num": expert_num,
                "Sensor": fbg,
                "Material": material,
                "Impact_Status": impact_status,
                "Methods_Detected": methods_detected,
                **feats
            }
            records.append(row)
            
            # Long format records for each feature
            for feat_key, feat_val in feats.items():
                unit = FEATURE_UNITS.get(feat_key, "")
                long_records.append({
                    "Sensor": fbg,
                    "Material": material,
                    "Experiment": expert_name,
                    "Feature": feat_key,
                    "Value": feat_val,
                    "Unit": unit,
                    "Impact_Status": impact_status
                })
                
    all_features_df = pd.DataFrame(records)
    long_format_df = pd.DataFrame(long_records)
    
    return all_features_df, long_format_df
