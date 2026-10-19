"""Input contracts and scientific invariants; no archived outputs required."""
import tempfile
import unittest
from unittest.mock import patch
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from run_pipeline import check_stage, main, validate_inputs
from src.phase_6.wavelet_analysis import extract_wavelet_features
from src.phase_6.fft_analysis import extract_fft_features


class PipelineTests(unittest.TestCase):
    def write_input(self, folder, time=None):
        time = np.arange(150) * 0.02 if time is None else time
        data = np.column_stack([time, np.ones((len(time), 4)), np.ones((len(time), 3)) * 1500])
        path = Path(folder) / "recording expert1.txt"
        path = path.with_name("recording expert1.001.txt")
        pd.DataFrame(data).to_csv(path, sep="\t", header=False, index=False)
        return path

    def test_valid_input_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_input(folder)
            files, records = validate_inputs(Path(folder))
            self.assertEqual(len(files), 1)
            self.assertEqual(records[0]["samples"], 150)
            self.assertEqual(len(records[0]["sha256"]), 64)

    def test_empty_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "No .txt"):
                validate_inputs(Path(folder))

    def test_duplicate_timestamps_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_input(folder, np.zeros(150))
            with self.assertRaisesRegex(ValueError, "timestamps"):
                validate_inputs(Path(folder))

    def test_nonfinite_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_input(folder)
            frame = pd.read_csv(path, sep="\t", header=None)
            frame.iloc[20, 5] = np.inf
            frame.to_csv(path, sep="\t", header=False, index=False)
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_inputs(Path(folder))

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_input(folder)
            output = Path(folder) / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep")
            self.assertEqual(main(["--input", folder, "--output", str(output), "--core-only"]), 1)
            self.assertEqual(marker.read_text(), "keep")

    def test_partial_phase3_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "data").mkdir()
            with self.assertRaisesRegex(RuntimeError, "every recording"):
                check_stage(Path(folder), "phase3_main.py", "data", 7)

    def test_wavelet_energy_and_entropy(self):
        result = extract_wavelet_features(np.sin(np.arange(128)))
        self.assertAlmostEqual(result["Wavelet_Energy"], result["Approximation_Energy"] + result["Detail_Energy"])
        self.assertGreaterEqual(result["Wavelet_Entropy"], 0)
        self.assertLessEqual(result["Wavelet_Entropy"], np.log2(5) + 1e-9)

    def test_fft_known_frequency(self):
        time = np.arange(500) / 50
        result = extract_fft_features(np.sin(2 * np.pi * 5 * time), 0.02)
        self.assertAlmostEqual(result["Dominant_Frequency"], 5)

    def test_failed_process_stops_pipeline_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_input(folder)
            output = Path(folder) / "run"

            def fake_process(command, **kwargs):
                if "stdout" in kwargs:
                    kwargs["stdout"].write("simulated stage failure\n")
                    return subprocess.CompletedProcess(command, 1)
                return subprocess.CompletedProcess(command, 0, stdout="test-environment\n")

            with patch("run_pipeline.subprocess.run", side_effect=fake_process):
                self.assertEqual(main(["--input", folder, "--output", str(output), "--core-only"]), 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(len(manifest["stages"]), 1)
            self.assertEqual(manifest["stages"][0]["status"], "failed")
            self.assertTrue((output / "logs/phase3_main.log").exists())

    def test_success_exit_with_failed_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_input(folder)
            output = Path(folder) / "run"

            def fake_process(command, **kwargs):
                if "stdout" in kwargs:
                    kwargs["stdout"].write("[CHECK 01] [FAIL] invalid output\n")
                return subprocess.CompletedProcess(command, 0, stdout="test\n")

            with patch("run_pipeline.subprocess.run", side_effect=fake_process):
                self.assertEqual(main(["--input", folder, "--output", str(output), "--core-only"]), 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(len(manifest["stages"]), 1)


if __name__ == "__main__":
    unittest.main()
