import numpy as np
from scipy.integrate import trapezoid


def calculate_baseline(signal_series, baseline_samples=100):
    """
    Calculate pre-impact baseline mean and noise standard deviation.
    
    Parameters:
        signal_series (pd.Series or np.ndarray): Raw or filtered FBG signal.
        baseline_samples (int): Number of initial samples used for baseline.
        
    Returns:
        tuple: (baseline_value, noise_std)
    """
    signal = np.asarray(signal_series)
    num_samples = min(len(signal), baseline_samples)
    baseline_window = signal[:num_samples]
    
    baseline_value = float(np.median(baseline_window))
    # Median Absolute Deviation (MAD) scaled for normal distribution
    mad = np.median(np.abs(baseline_window - baseline_value))
    noise_std = float(mad * 1.4826)
    if noise_std == 0 or np.isnan(noise_std):
        noise_std = float(np.std(baseline_window))
    if noise_std == 0:
        noise_std = 1e-6
        
    return baseline_value, noise_std


def calculate_peak_shift(signal_corrected, time):
    """
    1. Peak Shift
    Calculate maximum excursion relative to baseline.
    
    Returns:
        tuple: (peak_shift_signed, peak_shift_abs, peak_time, peak_idx)
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if len(signal) == 0:
        return np.nan, np.nan, np.nan, 0
        
    peak_idx = int(np.argmax(np.abs(signal)))
    peak_shift_signed = float(signal[peak_idx])
    peak_shift_abs = float(abs(peak_shift_signed))
    peak_time = float(t[peak_idx])
    
    return peak_shift_signed, peak_shift_abs, peak_time, peak_idx


def calculate_residual_shift(signal_corrected, time, recovery_end_idx=None, window_size=50):
    """
    2. Residual Shift
    Calculate remaining signal shift after recovery relative to pre-impact baseline.
    
    Returns:
        tuple: (residual_shift_signed, residual_shift_abs, post_recovery_level)
    """
    signal = np.asarray(signal_corrected)
    
    if len(signal) == 0:
        return np.nan, np.nan, np.nan
        
    if recovery_end_idx is not None and recovery_end_idx < len(signal) - 5:
        start_idx = recovery_end_idx
        end_idx = min(len(signal), recovery_end_idx + window_size)
    else:
        # Use final 15% of recorded samples if explicit recovery end not reached
        start_idx = max(0, int(len(signal) * 0.85))
        end_idx = len(signal)
        
    recovery_window = signal[start_idx:end_idx]
    if len(recovery_window) == 0:
        recovery_window = signal[-10:] if len(signal) >= 10 else signal
        
    residual_shift_signed = float(np.median(recovery_window))
    residual_shift_abs = float(abs(residual_shift_signed))
    post_recovery_level = residual_shift_signed
    
    return residual_shift_signed, residual_shift_abs, post_recovery_level


def calculate_rise_time(signal_corrected, time, peak_idx, impact_start_idx=0, low_pct=0.10, high_pct=0.90):
    """
    3. Rise Time
    Calculate time required to move from low_pct (10%) to high_pct (90%) of peak excursion.
    
    Returns:
        float: rise_time_seconds
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if peak_idx <= impact_start_idx or peak_idx >= len(signal):
        return np.nan
        
    peak_amp = abs(signal[peak_idx])
    if peak_amp == 0:
        return 0.0
        
    rising_segment = np.abs(signal[impact_start_idx:peak_idx + 1])
    rising_time = t[impact_start_idx:peak_idx + 1]
    
    target_low = low_pct * peak_amp
    target_high = high_pct * peak_amp
    
    # Find crossings
    idx_low = np.where(rising_segment >= target_low)[0]
    idx_high = np.where(rising_segment >= target_high)[0]
    
    if len(idx_low) == 0 or len(idx_high) == 0:
        return np.nan
        
    i_low = idx_low[0]
    i_high = idx_high[0]
    
    # Linear interpolation for precise timestamps
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
    
    Returns:
        tuple: (recovery_time_seconds, recovery_timestamp, recovery_end_idx)
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if peak_idx >= len(signal) - 1:
        return np.nan, np.nan, len(signal) - 1
        
    peak_time = t[peak_idx]
    peak_amp = abs(signal[peak_idx])
    
    # Recovery tolerance band: 20% of peak excursion or 3x noise_std, whichever is larger
    tolerance = max(3.0 * noise_std, 0.20 * peak_amp)
    
    post_peak_signal = signal[peak_idx:]
    post_peak_time = t[peak_idx:]
    
    within_tol = np.abs(post_peak_signal) <= tolerance
    
    recovery_rel_idx = None
    for i in range(len(within_tol) - confirmation_samples + 1):
        if np.all(within_tol[i:i + confirmation_samples]):
            recovery_rel_idx = i
            break
            
    if recovery_rel_idx is None:
        # If signal never stays within tolerance for full confirmation window, use last crossing
        last_cross = np.where(within_tol)[0]
        if len(last_cross) > 0:
            recovery_rel_idx = last_cross[-1]
        else:
            return np.nan, np.nan, len(signal) - 1
            
    recovery_end_idx = peak_idx + recovery_rel_idx
    recovery_timestamp = float(t[recovery_end_idx])
    recovery_time_seconds = max(0.0, float(recovery_timestamp - peak_time))
    
    return recovery_time_seconds, recovery_timestamp, recovery_end_idx


def calculate_peak_width(signal_corrected, time, peak_idx):
    """
    5. Peak Width (Full Width at Half Maximum - FWHM)
    Calculate temporal width of impact response at half-maximum excursion.
    
    Returns:
        float: peak_width_seconds
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if len(signal) < 3 or peak_idx < 0 or peak_idx >= len(signal):
        return np.nan
        
    peak_amp = abs(signal[peak_idx])
    half_max = 0.50 * peak_amp
    if half_max == 0:
        return 0.0
        
    abs_signal = np.abs(signal)
    
    # Left crossing (before peak)
    left_signal = abs_signal[:peak_idx + 1]
    left_time = t[:peak_idx + 1]
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
        t_half_left = t[0]
        
    # Right crossing (after peak)
    right_signal = abs_signal[peak_idx:]
    right_time = t[peak_idx:]
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
        t_half_right = t[-1]
        
    peak_width_seconds = max(0.0, float(t_half_right - t_half_left))
    return peak_width_seconds


def calculate_max_slope(signal_corrected, time):
    """
    6. Maximum Slope
    Calculate maximum rate of signal change max(|dSignal/dt|).
    
    Returns:
        tuple: (max_slope_pos, max_slope_neg, max_slope_abs)
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if len(signal) < 2:
        return np.nan, np.nan, np.nan
        
    dt = np.gradient(t)
    dt[dt == 0] = 1e-6
    ds = np.gradient(signal)
    slope = ds / dt
    
    max_slope_pos = float(np.max(slope))
    max_slope_neg = float(np.min(slope))
    max_slope_abs = float(np.max(np.abs(slope)))
    
    return max_slope_pos, max_slope_neg, max_slope_abs


def calculate_rms(signal_corrected):
    """
    7. RMS
    Calculate Root Mean Square of baseline-corrected signal.
    Formula: RMS = sqrt(mean(x^2))
    
    Returns:
        float: rms_val
    """
    signal = np.asarray(signal_corrected)
    if len(signal) == 0:
        return np.nan
    return float(np.sqrt(np.mean(signal ** 2)))


def calculate_signal_energy(signal_corrected, time):
    """
    8. Signal Energy
    Calculate impact signal energy: Energy = integral x(t)^2 dt
    
    Returns:
        float: signal_energy
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if len(signal) < 2:
        return np.nan
    energy = float(trapezoid(signal ** 2, x=t))
    return energy


def calculate_peak_to_peak(signal_corrected):
    """
    9. Peak-to-Peak
    Calculate Peak-to-Peak amplitude: max(x) - min(x)
    
    Returns:
        float: peak_to_peak
    """
    signal = np.asarray(signal_corrected)
    if len(signal) == 0:
        return np.nan
    return float(np.max(signal) - np.min(signal))


def calculate_variance(signal_corrected, ddof=1):
    """
    10. Variance
    Calculate variance of baseline-corrected signal.
    
    Returns:
        float: variance
    """
    signal = np.asarray(signal_corrected)
    if len(signal) <= ddof:
        return np.nan
    return float(np.var(signal, ddof=ddof))


def calculate_std(signal_corrected, ddof=1):
    """
    11. Standard Deviation
    Calculate standard deviation of baseline-corrected signal.
    
    Returns:
        float: std_dev
    """
    signal = np.asarray(signal_corrected)
    if len(signal) <= ddof:
        return np.nan
    return float(np.std(signal, ddof=ddof))


def calculate_entropy(signal_corrected, bins="fd"):
    """
    12. Entropy
    Calculate histogram-based Shannon entropy H = -sum p log2(p).
    
    Returns:
        float: entropy_bits
    """
    signal = np.asarray(signal_corrected)
    if len(signal) < 2:
        return 0.0
        
    counts, _ = np.histogram(signal, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
        
    probs = counts / total
    probs = probs[probs > 0]
    entropy_bits = float(-np.sum(probs * np.log2(probs)))
    return entropy_bits


def calculate_auc(signal_corrected, time):
    """
    13. Area Under Curve (AUC)
    Calculate signed AUC integral x(t) dt and absolute AUC integral |x(t)| dt.
    
    Returns:
        tuple: (auc_signed, auc_abs)
    """
    signal = np.asarray(signal_corrected)
    t = np.asarray(time)
    
    if len(signal) < 2:
        return np.nan, np.nan
        
    auc_signed = float(trapezoid(signal, x=t))
    auc_abs = float(trapezoid(np.abs(signal), x=t))
    
    return auc_signed, auc_abs
