import numpy as np

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

def detect_threshold(time_series, signal_series):
    signal = signal_series.values
    time = time_series.values
    
    b_mean, b_noise, start_idx = get_stable_baseline(signal, window_size=50)
        
    # Requires an excursion of at least 5x MAD
    threshold_margin = max(5.0 * b_noise, 1e-4)
    
    is_above = np.abs(signal - b_mean) > threshold_margin
    
    # 3 consecutive samples beyond 5x MAD is extremely robust against isolated noise spikes
    persistence = 3
    for i in range(start_idx, len(is_above) - persistence + 1):
        if np.all(is_above[i:i+persistence]):
            signal_val = signal[i]
            dev_noise_ratio = abs(signal_val - b_mean) / b_noise if b_noise > 0 else 0
            return True, time[i], b_mean, b_noise, threshold_margin, signal_val, dev_noise_ratio
            
    return False, None, b_mean, b_noise, threshold_margin, None, None
