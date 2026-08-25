import numpy as np
from scipy.signal import find_peaks

def get_stable_baseline(signal, window_size=50):
    search_end = max(int(len(signal) * 0.2), window_size * 2)
    search_end = min(search_end, len(signal))
    
    if search_end <= window_size:
        window = signal
        b_mean = np.median(window)
        b_noise = np.median(np.abs(window - b_mean)) * 1.4826
        return b_mean, b_noise, len(window)
        
    min_std = float('inf')
    best_start = 0
    
    for i in range(search_end - window_size):
        window = signal[i:i+window_size]
        w_std = np.std(window)
        if w_std < min_std:
            min_std = w_std
            best_start = i
            
    baseline = signal[best_start:best_start+window_size]
    b_mean = np.median(baseline)
    b_noise = np.median(np.abs(baseline - b_mean)) * 1.4826
    if b_noise == 0:
        b_noise = 1e-6
    return b_mean, b_noise, search_end

def detect_peak(time_series, signal_series):
    signal = signal_series.values
    time = time_series.values
    
    b_mean, b_noise, start_idx = get_stable_baseline(signal, window_size=50)
        
    # Since we now have a genuinely quiet baseline, we need a robust multiplier
    # to avoid triggering on random noise over 5000+ samples.
    min_prominence = max(5.0 * b_noise, 1e-4)
    min_deviation = max(6.0 * b_noise, 1e-4)
    
    peaks_pos, props_pos = find_peaks(signal, prominence=min_prominence, width=1)
    peaks_neg, props_neg = find_peaks(-signal, prominence=min_prominence, width=1)
    
    valid_peaks = []
    
    for p in peaks_pos:
        if p >= start_idx and abs(signal[p] - b_mean) > min_deviation:
            valid_peaks.append(p)
            
    for p in peaks_neg:
        if p >= start_idx and abs(signal[p] - b_mean) > min_deviation:
            valid_peaks.append(p)
            
    if valid_peaks:
        first_peak = min(valid_peaks)
        signal_val = signal[first_peak]
        dev_noise_ratio = abs(signal_val - b_mean) / b_noise if b_noise > 0 else 0
        return True, time[first_peak], b_mean, b_noise, min_deviation, signal_val, dev_noise_ratio
        
    return False, None, b_mean, b_noise, min_deviation, None, None
