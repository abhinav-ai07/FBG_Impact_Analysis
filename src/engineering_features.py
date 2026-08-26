import numpy as np
from scipy.integrate import trapezoid


def calculate_baseline(signal_series, baseline_samples=100):
    """
    Calculate pre-impact baseline mean and noise standard deviation.
    
    Parameters:
        signal_series (pd.Series or np.ndarray): FBG signal.
        baseline_samples (int): Initial samples used for baseline.
        
    Returns:
        tuple: (baseline_value, noise_std)
    """
    signal = np.asarray(signal_series)
    num_samples = min(len(signal), baseline_samples)
    baseline_window = signal[:num_samples]
    
    baseline_value = float(np.median(baseline_window))
    mad = np.median(np.abs(baseline_window - baseline_value))
    noise_std = float(mad * 1.4826)
    if noise_std == 0 or np.isnan(noise_std):
        noise_std = float(np.std(baseline_window))
    if noise_std == 0:
        noise_std = 1e-6
        
    return baseline_value, noise_std


def calculate_peak_shift(signal_corrected, time, peak_idx):
    """
    1. Peak Shift
    Calculate maximum wavelength shift at the detected Phase 4 peak index.
    
    Returns:
        tuple: (peak_shift_signed, peak_shift_abs, peak_time)
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if peak_idx is None or peak_idx < 0 or peak_idx >= len(signal):
        return np.nan, np.nan, np.nan
        
    peak_shift_signed = float(signal[peak_idx])
    peak_shift_abs = float(abs(peak_shift_signed))
    peak_time = float(t[peak_idx])
    
    return peak_shift_signed, peak_shift_abs, peak_time


def calculate_residual_shift(signal_corrected, time, recovery_end_idx, window_size=50):
    """
    2. Residual Shift
    Calculate remaining signal shift after confirmed recovery relative to baseline.
    If recovery is not confirmed, returns NaN.
    
    Returns:
        tuple: (residual_shift_signed, residual_shift_abs, post_recovery_level)
    """
    signal = np.asarray(signal_corrected)
    
    if recovery_end_idx is None or recovery_end_idx >= len(signal):
        return np.nan, np.nan, np.nan
        
    start_idx = recovery_end_idx
    end_idx = min(len(signal), recovery_end_idx + window_size)
    recovery_window = signal[start_idx:end_idx]
    
    if len(recovery_window) == 0:
        return np.nan, np.nan, np.nan
        
    residual_shift_signed = float(np.median(recovery_window))
    residual_shift_abs = float(abs(residual_shift_signed))
    post_recovery_level = residual_shift_signed
    
    return residual_shift_signed, residual_shift_abs, post_recovery_level


def calculate_rise_time(signal_corrected, time, event_start_idx, peak_idx, low_pct=0.10, high_pct=0.90):
    """
    3. Rise Time
    Calculate time required to move from 10% to 90% of peak excursion between event start and peak.
    If not determinable, returns NaN.
    
    Returns:
        float: rise_time_seconds
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if event_start_idx is None or peak_idx is None or peak_idx <= event_start_idx or peak_idx >= len(signal):
        return np.nan
        
    peak_amp = abs(signal[peak_idx])
    if peak_amp == 0:
        return np.nan
        
    rising_segment = np.abs(signal[event_start_idx:peak_idx + 1])
    rising_time = t[event_start_idx:peak_idx + 1]
    
    target_low = low_pct * peak_amp
    target_high = high_pct * peak_amp
    
    idx_low = np.where(rising_segment >= target_low)[0]
    idx_high = np.where(rising_segment >= target_high)[0]
    
    if len(idx_low) == 0 or len(idx_high) == 0:
        return np.nan
        
    i_low = idx_low[0]
    i_high = idx_high[0]
    
    if i_low > 0:
        s0, s1 = rising_segment[i_low - 1], rising_segment[i_low]
        t0, t1 = rising_time[i_low - 1], rising_time[i_low]
        t_low = t0 + (target_low - s0) * (t1 - t0) / (s1 - s0) if s1 != s0 else t0
    else:
        t_low = rising_time[0]
        
    if i_high > 0:
        s0, s1 = rising_segment[i_high - 1], rising_segment[i_high]
        t0, t1 = rising_time[i_high - 1], rising_time[i_high]
        t_high = t0 + (target_high - s0) * (t1 - t0) / (s1 - s0) if s1 != s0 else t0
    else:
        t_high = rising_time[i_high]
        
    rise_time_seconds = max(0.0, float(t_high - t_low))
    return rise_time_seconds


def calculate_recovery_time(signal_corrected, time, peak_idx, noise_std, confirmation_samples=5):
    """
    4. Recovery Time
    Calculate time required after peak for signal to return to tolerance band around baseline.
    If the signal does not return and remain within tolerance before recording ends, returns NaN.
    
    Returns:
        tuple: (recovery_time_seconds, recovery_timestamp, recovery_end_idx, recovery_confirmed)
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if peak_idx is None or peak_idx >= len(signal) - 5:
        return np.nan, np.nan, None, False
        
    peak_time = t[peak_idx]
    peak_amp = abs(signal[peak_idx])
    
    tolerance = max(3.0 * noise_std, 0.20 * peak_amp)
    
    post_peak_signal = signal[peak_idx:]
    post_peak_time = t[peak_idx:]
    
    within_tol = np.abs(post_peak_signal) <= tolerance
    
    recovery_rel_idx = None
    for i in range(len(within_tol) - confirmation_samples + 1):
        if np.all(within_tol[i:i + confirmation_samples]):
            recovery_rel_idx = i
            break
            
    # Check if recovery occurred before end of recording (at least 10 samples margin from end)
    if recovery_rel_idx is None or (peak_idx + recovery_rel_idx) >= len(signal) - 10:
        return np.nan, np.nan, None, False
        
    recovery_end_idx = peak_idx + recovery_rel_idx
    recovery_timestamp = float(t[recovery_end_idx])
    recovery_time_seconds = max(0.0, float(recovery_timestamp - peak_time))
    
    return recovery_time_seconds, recovery_timestamp, recovery_end_idx, True


def calculate_peak_width(signal_corrected, time, peak_idx, event_start_idx=0, recovery_end_idx=None):
    """
    5. Peak Width (Full Width at Half Maximum - FWHM)
    Calculate temporal width of impact response at half-maximum excursion within the event.
    
    Returns:
        float: peak_width_seconds
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if peak_idx is None or peak_idx < 0 or peak_idx >= len(signal):
        return np.nan
        
    peak_amp = abs(signal[peak_idx])
    half_max = 0.50 * peak_amp
    if half_max == 0:
        return np.nan
        
    abs_signal = np.abs(signal)
    
    end_bound = recovery_end_idx if recovery_end_idx is not None else len(signal) - 1
    start_bound = event_start_idx if event_start_idx is not None else 0
    
    # Left crossing
    left_signal = abs_signal[start_bound:peak_idx + 1]
    left_time = t[start_bound:peak_idx + 1]
    left_above = np.where(left_signal >= half_max)[0]
    
    if len(left_above) > 0:
        i_left = left_above[0]
        if i_left > 0:
            s0, s1 = left_signal[i_left - 1], left_signal[i_left]
            t0, t1 = left_time[i_left - 1], left_time[i_left]
            t_half_left = t0 + (half_max - s0) * (t1 - t0) / (s1 - s0) if s1 != s0 else t0
        else:
            t_half_left = left_time[0]
    else:
        return np.nan
        
    # Right crossing
    right_signal = abs_signal[peak_idx:end_bound + 1]
    right_time = t[peak_idx:end_bound + 1]
    right_below = np.where(right_signal < half_max)[0]
    
    if len(right_below) > 0:
        i_right = right_below[0]
        if i_right > 0:
            s0, s1 = right_signal[i_right - 1], right_signal[i_right]
            t0, t1 = right_time[i_right - 1], right_time[i_right]
            t_half_right = t0 + (half_max - s0) * (t1 - t0) / (s1 - s0) if s1 != s0 else t0
        else:
            t_half_right = right_time[0]
    else:
        return np.nan
        
    peak_width_seconds = max(0.0, float(t_half_right - t_half_left))
    return peak_width_seconds


def calculate_max_slope(signal_corrected, time, event_start_idx, event_end_idx):
    """
    6. Maximum Slope
    Calculate max(|dSignal/dt|) within the Phase 4 event boundary.
    
    Returns:
        tuple: (max_slope_pos, max_slope_neg, max_slope_abs)
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if event_start_idx is None or event_end_idx is None or event_end_idx <= event_start_idx:
        return np.nan, np.nan, np.nan
        
    seg_sig = signal[event_start_idx:event_end_idx + 1]
    seg_t = t[event_start_idx:event_end_idx + 1]
    
    if len(seg_sig) < 2:
        return np.nan, np.nan, np.nan
        
    dt = np.gradient(seg_t)
    dt[dt == 0] = 1e-6
    ds = np.gradient(seg_sig)
    slope = ds / dt
    
    max_slope_pos = float(np.max(slope))
    max_slope_neg = float(np.min(slope))
    max_slope_abs = float(np.max(np.abs(slope)))
    
    return max_slope_pos, max_slope_neg, max_slope_abs


def calculate_rms(signal_corrected, event_start_idx, event_end_idx):
    """
    7. RMS
    Calculate RMS over Phase 4 event window: sqrt(mean(x^2)).
    """
    signal = np.asarray(signal_corrected)
    if event_start_idx is None or event_end_idx is None or event_end_idx <= event_start_idx:
        return np.nan
    window = signal[event_start_idx:event_end_idx + 1]
    if len(window) == 0:
        return np.nan
    return float(np.sqrt(np.mean(window ** 2)))


def calculate_signal_energy(signal_corrected, time, event_start_idx, event_end_idx):
    """
    8. Signal Energy
    Calculate signal energy over Phase 4 event window: integral x(t)^2 dt.
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    if event_start_idx is None or event_end_idx is None or event_end_idx <= event_start_idx:
        return np.nan
    win_sig = signal[event_start_idx:event_end_idx + 1]
    win_t = t[event_start_idx:event_end_idx + 1]
    if len(win_sig) < 2:
        return np.nan
    return float(trapezoid(win_sig ** 2, x=win_t))


def calculate_peak_to_peak(signal_corrected, event_start_idx, event_end_idx):
    """
    9. Peak-to-Peak
    Calculate max(x) - min(x) over Phase 4 event window.
    """
    signal = np.asarray(signal_corrected)
    if event_start_idx is None or event_end_idx is None or event_end_idx <= event_start_idx:
        return np.nan
    window = signal[event_start_idx:event_end_idx + 1]
    if len(window) == 0:
        return np.nan
    return float(np.max(window) - np.min(window))


def calculate_variance(signal_corrected, event_start_idx, event_end_idx, ddof=1):
    """
    10. Variance
    Calculate sample variance over Phase 4 event window.
    """
    signal = np.asarray(signal_corrected)
    if event_start_idx is None or event_end_idx is None or event_end_idx <= event_start_idx:
        return np.nan
    window = signal[event_start_idx:event_end_idx + 1]
    if len(window) <= ddof:
        return np.nan
    return float(np.var(window, ddof=ddof))


def calculate_std(signal_corrected, event_start_idx, event_end_idx, ddof=1):
    """
    11. Standard Deviation
    Calculate sample standard deviation over Phase 4 event window.
    """
    signal = np.asarray(signal_corrected)
    if event_start_idx is None or event_end_idx is None or event_end_idx <= event_start_idx:
        return np.nan
    window = signal[event_start_idx:event_end_idx + 1]
    if len(window) <= ddof:
        return np.nan
    return float(np.std(window, ddof=ddof))


def calculate_entropy(signal_corrected, event_start_idx, event_end_idx, bins="fd"):
    """
    12. Entropy
    Calculate histogram-based Shannon entropy over Phase 4 event window.
    """
    signal = np.asarray(signal_corrected)
    if event_start_idx is None or event_end_idx is None or event_end_idx <= event_start_idx:
        return np.nan
    window = signal[event_start_idx:event_end_idx + 1]
    if len(window) < 2:
        return np.nan
    counts, _ = np.histogram(window, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def calculate_auc(signal_corrected, time, event_start_idx, event_end_idx):
    """
    13. Area Under Curve (AUC)
    Calculate signed AUC and absolute AUC over Phase 4 event window.
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    if event_start_idx is None or event_end_idx is None or event_end_idx <= event_start_idx:
        return np.nan, np.nan
    win_sig = signal[event_start_idx:event_end_idx + 1]
    win_t = t[event_start_idx:event_end_idx + 1]
    if len(win_sig) < 2:
        return np.nan, np.nan
    auc_signed = float(trapezoid(win_sig, x=win_t))
    auc_abs = float(trapezoid(np.abs(win_sig), x=win_t))
    return auc_signed, auc_abs
