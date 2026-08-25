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

def detect_changepoint(time_series, signal_series):
    signal = signal_series.values
    time = time_series.values
    
    window_size = max(int(len(signal) * 0.02), 10)
    
    b_mean, b_noise, start_idx = get_stable_baseline(signal, window_size=50)
        
    # Real changepoint in an impact event will cause a massive variance increase
    # or a persistent massive mean shift.
    std_threshold_ratio = 5.0
    mean_shift_threshold_ratio = 5.0
    
    std_threshold = max(std_threshold_ratio * b_noise, 1e-4)
    mean_shift_threshold = max(mean_shift_threshold_ratio * b_noise, 1e-4)
    
    for i in range(start_idx, len(signal) - window_size):
        window = signal[i:i+window_size]
        w_mean = np.median(window)
        w_std = np.median(np.abs(window - w_mean)) * 1.4826
        
        is_mean_shift = abs(w_mean - b_mean) > mean_shift_threshold
        is_std_shift = w_std > std_threshold
        
        if is_std_shift or is_mean_shift:
            # Check post-change persistence
            persistence = window_size
            if i + persistence < len(signal):
                future_window = signal[i:i+persistence]
                f_mean = np.median(future_window)
                f_std = np.median(np.abs(future_window - f_mean)) * 1.4826
                
                f_is_mean_shift = abs(f_mean - b_mean) > mean_shift_threshold
                f_is_std_shift = f_std > std_threshold
                
                if (is_mean_shift and f_is_mean_shift) or (is_std_shift and f_is_std_shift):
                    
                    if is_mean_shift and f_is_mean_shift:
                        signal_val = w_mean
                        dev_noise_ratio = abs(w_mean - b_mean) / b_noise if b_noise > 0 else 0
                        return True, time[i], b_mean, b_noise, mean_shift_threshold, signal_val, dev_noise_ratio
                    else:
                        signal_val = w_std
                        dev_noise_ratio = w_std / b_noise if b_noise > 0 else 0
                        return True, time[i], b_noise, b_noise, std_threshold, signal_val, dev_noise_ratio
                    
    return False, None, b_mean, b_noise, mean_shift_threshold, None, None
