import numpy as np
from scipy.fft import fft, fftfreq


def extract_fft_features(signal, sampling_interval):

    signal = np.asarray(
        signal,
        dtype=float
    )

    N = len(signal)

    fft_values = np.abs(
        fft(signal)
    )

    freqs = fftfreq(
        N,
        d=sampling_interval
    )

    positive = freqs > 0

    freqs = freqs[positive]
    fft_values = fft_values[positive]

    dominant_frequency = freqs[
        np.argmax(fft_values)
    ]

    spectral_energy = np.sum(
        fft_values ** 2
    )

    prob = fft_values / (
        np.sum(fft_values)
        + 1e-12
    )

    spectral_entropy = -np.sum(
        prob *
        np.log2(prob + 1e-12)
    )

    spectral_centroid = (
        np.sum(
            freqs * fft_values
        )
        /
        (
            np.sum(fft_values)
            + 1e-12
        )
    )

    bandwidth = np.sqrt(
        np.sum(
            (
                (freqs - spectral_centroid)
                ** 2
            )
            * fft_values
        )
        /
        (
            np.sum(fft_values)
            + 1e-12
        )
    )

    return {
        "Dominant_Frequency": dominant_frequency,
        "Spectral_Energy": spectral_energy,
        "Spectral_Entropy": spectral_entropy,
        "Spectral_Centroid": spectral_centroid,
        "Bandwidth": bandwidth
    }