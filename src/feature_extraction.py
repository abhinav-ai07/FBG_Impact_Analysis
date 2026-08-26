import os
import glob
import pandas as pd
import numpy as np

from peak_detection import detect_peak
from threshold_detection import detect_threshold
from derivative_detection import detect_derivative
from changepoint_detection import detect_changepoint

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

# Material mapping for FBG channels (STRICTLY CONSISTENT)
MATERIAL_MAP = {
    "FBG1": "Copper",
    "FBG2": "Bare",
    "FBG3": "Steel"
}

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
    Extract Phase 5 engineering features for a single FBG signal.
    
    If impact_detected is False (NO IMPACT case), returns NaN for all impact-event features.
    """
    time = np.asarray(time_series)
    signal = np.asarray(signal_series)
    
    # Baseline calculation
    baseline_val, noise_std = calculate_baseline(signal, baseline_samples=baseline_samples)
    signal_corrected = signal - baseline_val
    
    # ----------------------------------------------------
    # NO IMPACT CASE: Return NaN for event-based features
    # ----------------------------------------------------
    if not impact_detected:
        return {
            "baseline_nm": baseline_val,
            "noise_std_nm": noise_std,
            "peak_shift_signed": np.nan,
            "peak_shift_abs": np.nan,
            "peak_time": np.nan,
            "residual_shift_signed": np.nan,
            "residual_shift_abs": np.nan,
            "recovered_level": np.nan,
            "rise_time_seconds": np.nan,
            "recovery_time_seconds": np.nan,
            "recovery_timestamp": np.nan,
            "peak_width_seconds": np.nan,
            "max_slope_pos": np.nan,
            "max_slope_neg": np.nan,
            "max_slope_abs": np.nan,
            "rms": np.nan,
            "signal_energy": np.nan,
            "peak_to_peak": np.nan,
            "variance": np.nan,
            "std_dev": np.nan,
            "entropy": np.nan,
            "auc_signed": np.nan,
            "auc_abs": np.nan
        }
        
    # ----------------------------------------------------
    # IMPACT CASE: Use Phase 4 detected event boundaries
    # ----------------------------------------------------
    p_res, p_t, _, _, _, _, _ = detect_peak(time_series, signal_series)
    t_res, t_t, _, _, _, _, _ = detect_threshold(time_series, signal_series)
    d_res, d_t, _, _, _, _, _ = detect_derivative(time_series, signal_series)
    c_res, c_t, _, _, _, _, _ = detect_changepoint(time_series, signal_series)
    
    triggers = [tm for res, tm in [(p_res, p_t), (t_res, t_t), (d_res, d_t), (c_res, c_t)] if res and tm is not None]
    
    if not triggers:
        # Fallback if no specific method timestamp was captured
        event_start_idx = 0
    else:
        earliest_trigger = min(triggers)
        event_start_idx = int(np.searchsorted(time, earliest_trigger))
        
    # Peak index within Phase 4 event window (next 200 samples / 4 seconds)
    search_end = min(len(signal_corrected), event_start_idx + 200)
    event_segment = signal_corrected[event_start_idx:search_end]
    if len(event_segment) > 0:
        rel_peak_idx = int(np.argmax(np.abs(event_segment)))
        peak_idx = event_start_idx + rel_peak_idx
    else:
        peak_idx = event_start_idx
        
    # 1. Peak Shift
    peak_shift_signed, peak_shift_abs, peak_time = calculate_peak_shift(signal_corrected, time, peak_idx)
    
    # 2. Recovery Time
    recovery_time_seconds, recovery_timestamp, recovery_end_idx, rec_confirmed = calculate_recovery_time(
        signal_corrected, time, peak_idx, noise_std
    )
    
    # Event end boundary
    event_end_idx = recovery_end_idx if (rec_confirmed and recovery_end_idx is not None) else min(len(signal_corrected) - 1, peak_idx + 150)
    
    # 3. Residual Shift
    residual_shift_signed, residual_shift_abs, recovered_level = calculate_residual_shift(
        signal_corrected, time, recovery_end_idx if rec_confirmed else None
    )
    
    # 4. Rise Time
    rise_time_seconds = calculate_rise_time(signal_corrected, time, event_start_idx, peak_idx)
    
    # 5. Peak Width (FWHM)
    peak_width_seconds = calculate_peak_width(signal_corrected, time, peak_idx, event_start_idx, event_end_idx)
    
    # 6. Maximum Slope
    max_slope_pos, max_slope_neg, max_slope_abs = calculate_max_slope(signal_corrected, time, event_start_idx, event_end_idx)
    
    # 7. RMS
    rms_val = calculate_rms(signal_corrected, event_start_idx, event_end_idx)
    
    # 8. Signal Energy
    signal_energy = calculate_signal_energy(signal_corrected, time, event_start_idx, event_end_idx)
    
    # 9. Peak-to-Peak
    peak_to_peak = calculate_peak_to_peak(signal_corrected, event_start_idx, event_end_idx)
    
    # 10. Variance
    variance_val = calculate_variance(signal_corrected, event_start_idx, event_end_idx)
    
    # 11. Standard Deviation
    std_val = calculate_std(signal_corrected, event_start_idx, event_end_idx)
    
    # 12. Entropy
    entropy_bits = calculate_entropy(signal_corrected, event_start_idx, event_end_idx)
    
    # 13. Area Under Curve (AUC)
    auc_signed, auc_abs = calculate_auc(signal_corrected, time, event_start_idx, event_end_idx)
    
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
    Extract Phase 5 features across all datasets using verified Phase 4 detection results.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    def extract_expert_num(filename):
        base = os.path.basename(filename)
        try:
            return int(base.split("expert")[1].split(".")[0])
        except (IndexError, ValueError):
            return 0
            
    csv_files.sort(key=extract_expert_num)
    
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
            
            impact_status = "NO IMPACT"
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
