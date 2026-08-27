import os
import numpy as np
import pandas as pd

from src.phase_6.fft_analysis import (
    extract_fft_features
)

from src.phase_6.wavelet_analysis import (
    extract_wavelet_features
)


INPUT_DIR = r"data\processed\final_phase_input"

OUTPUT_DIR = r"results\phase6"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

all_results = []

print("\nPHASE 6 STARTED\n")


for filename in os.listdir(INPUT_DIR):

    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(
        INPUT_DIR,
        filename
    )

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

    for sensor in sensors:

        signal = df[sensor].values

        fft_features = (
            extract_fft_features(
                signal,
                sampling_interval
            )
        )

        wavelet_features = (
            extract_wavelet_features(
                signal
            )
        )

        row = {
            "File": filename,
            "Sensor": sensor
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

print("\nFirst 5 Results:\n")

print(
    results_df.head()
)