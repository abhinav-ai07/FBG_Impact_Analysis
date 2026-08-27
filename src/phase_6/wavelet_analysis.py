import numpy as np
import pywt


def extract_wavelet_features(signal):

    signal = np.asarray(signal, dtype=float)

    coeffs = pywt.wavedec(
        signal,
        wavelet="db4",
        level=4
    )

    approx_energy = np.sum(
        coeffs[0] ** 2
    )

    detail_energy = np.sum(
        [
            np.sum(c ** 2)
            for c in coeffs[1:]
        ]
    )

    wavelet_energy = (
        approx_energy +
        detail_energy
    )

    energies = np.array(
        [
            np.sum(c ** 2)
            for c in coeffs
        ]
    )

    prob = energies / (
        np.sum(energies) + 1e-12
    )

    wavelet_entropy = -np.sum(
        prob *
        np.log2(prob + 1e-12)
    )

    detail_approx_ratio = (
        detail_energy /
        (approx_energy + 1e-12)
    )

    return {
        "Approximation_Energy": approx_energy,
        "Detail_Energy": detail_energy,
        "Wavelet_Energy": wavelet_energy,
        "Wavelet_Entropy": wavelet_entropy,
        "Detail_Approx_Ratio": detail_approx_ratio
    }