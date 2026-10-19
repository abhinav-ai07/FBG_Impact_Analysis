"""Run all analysis stages in a fresh workspace without modifying archived results."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("phase3_main.py", "data/processed/final_phase_input"),
    ("phase4.py", "results/phase4/phase4_method_results.csv"),
    ("phase5.py", "results/phase5/phase5_all_features.csv"),
    ("phase6.py", "results/phase6/phase6_multidomain_features.csv"),
    ("phase7.py", "results/phase7/phase7_statistical_summary.csv"),
    ("phase8.py", "results/phase8/phase8_engineering_indices.csv"),
    ("phase9_pca.py", "results/phase9/pca/phase9_pca_scores.csv"),
    ("phase9_umap.py", "results/phase9/umap/phase9_umap_complete_multidomain_coordinates.csv"),
    ("phase9_part2.py", "results/phase9/phase9_part2_summary.md"),
]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(source):
    import numpy as np
    import pandas as pd
    files = sorted(source.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt recordings found in {source}")
    experts = set()
    records = []
    for path in files:
        match = re.search(r"expert(\d+)\.", path.name, re.I)
        if not match or int(match[1]) in experts:
            raise ValueError(f"Use unique filenames containing expert<number>.: {path.name}")
        experts.add(int(match[1]))
        frame = pd.read_csv(path, sep="\t", header=None)
        if frame.shape[1] != 8 or len(frame) < 110:
            raise ValueError(f"{path.name}: expected 8 columns and at least 110 samples")
        values = frame.iloc[:, [0, 5, 6, 7]].apply(pd.to_numeric, errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: time and sensor readings must be finite numbers")
        intervals = np.diff(values[:, 0])
        if not (intervals > 0).all() or not np.allclose(intervals, 0.02, atol=1e-5, rtol=0.01):
            raise ValueError(f"{path.name}: expected increasing, uniformly sampled 50 Hz timestamps")
        records.append({"name": path.name, "samples": len(frame), "sha256": fingerprint(path)})
    return files, records


def check_stage(workspace, script, artifact, recordings):
    import numpy as np
    import pandas as pd
    path = workspace / artifact
    if not path.exists():
        raise RuntimeError(f"{script}: missing output {artifact}")
    if script == "phase3_main.py":
        if len(list(path.glob("*.csv"))) != recordings:
            raise RuntimeError("Phase 3 did not process every recording")
        return
    if path.suffix != ".csv":
        return
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"{script}: empty output")
    if script in {"phase4.py", "phase5.py", "phase6.py", "phase8.py"}:
        if len(frame) != recordings * 3:
            raise RuntimeError(f"{script}: expected {recordings * 3} sensor cases, got {len(frame)}")
        keys = {"phase4.py": ["Dataset", "FBG"], "phase5.py": ["Dataset", "Sensor"],
                "phase6.py": ["File", "Sensor"], "phase8.py": ["Expert", "FBG"]}[script]
        if frame.duplicated(keys).any():
            raise RuntimeError(f"{script}: duplicate sensor cases")
    if script == "phase6.py":
        features = frame.columns[3:]
        impact = frame.Phase4_Result.eq("IMPACT")
        if not np.isfinite(frame.loc[impact, features].to_numpy(dtype=float)).all():
            raise RuntimeError("Phase 6 has invalid impact features")
        if not frame.loc[~impact, features].isna().all().all():
            raise RuntimeError("Phase 6 computed event features for NO IMPACT cases")
    if script == "phase8.py":
        scores = frame.loc[frame.Impact_Status.eq("IMPACT"), ["DSTI", "PEI", "SII", "DRI"]]
        if not np.isfinite(scores.to_numpy()).all() or not ((scores >= 0) & (scores <= 1)).all().all():
            raise RuntimeError("Phase 8 has invalid engineering indices")
    if script in {"phase5.py", "phase6.py", "phase8.py"}:
        original = pd.read_csv(workspace / "results/phase4/phase4_method_results.csv")
        expected = original.set_index(["Dataset", "FBG"])["Final_Result"].sort_index()
        observed = frame.copy()
        if script == "phase5.py":
            observed = observed.rename(columns={"Sensor": "FBG", "Impact_Status": "Decision"})
        elif script == "phase6.py":
            observed["Dataset"] = "Expert " + observed.File.str.extract(r"expert(\d+)", expand=False)
            observed["FBG"] = observed.Sensor.str.replace("_processed", "", regex=False)
            observed = observed.rename(columns={"Phase4_Result": "Decision"})
        else:
            observed = observed.rename(columns={"Expert": "Dataset", "Impact_Status": "Decision"})
        actual = observed.set_index(["Dataset", "FBG"])["Decision"].sort_index()
        if not expected.equals(actual.rename(expected.name)):
            raise RuntimeError(f"{script}: sensor identities or decisions differ from Phase 4")
    if script in {"phase9_pca.py", "phase9_umap.py"}:
        previous = pd.read_csv(workspace / "results/phase5/phase5_all_features.csv")
        expected = previous[previous.Impact_Status.eq("IMPACT")]
        expected_keys = set(zip(expected.Dataset, expected.Sensor))
        if frame.duplicated(["Expert", "FBG"]).any() or set(zip(frame.Expert, frame.FBG)) != expected_keys:
            raise RuntimeError(f"{script}: embedding cases differ from upstream impact cases")


def write_report(workspace, manifest):
    import pandas as pd
    decisions = pd.read_csv(workspace / "results/phase4/phase4_method_results.csv")
    indices = pd.read_csv(workspace / "results/phase8/phase8_engineering_indices.csv")
    lines = ["# FBG analysis run", "", f"Status: **{manifest['status']}**", "",
             f"Recordings: {len(manifest['inputs'])}; sensor cases: {len(decisions)}.", "",
             "## Impact decisions", "", "| Material | IMPACT | NO IMPACT |", "|---|---:|---:|"]
    for sensor, material in {"FBG1": "Copper", "FBG2": "Bare", "FBG3": "Steel"}.items():
        subset = decisions[decisions.FBG.eq(sensor)]
        count = int(subset.Final_Result.eq("IMPACT").sum())
        lines.append(f"| {material} | {count} | {len(subset) - count} |")
    lines += ["", "## Mean engineering indices (IMPACT only)", "",
              "| Material | DSTI | PEI | SII | DRI |", "|---|---:|---:|---:|---:|"]
    for material, group in indices[indices.Impact_Status.eq("IMPACT")].groupby("Material"):
        means = group[["DSTI", "PEI", "SII", "DRI"]].mean()
        lines.append(f"| {material} | " + " | ".join(f"{x:.4f}" for x in means) + " |")
    if (workspace / "results/phase9/pca/phase9_pca_explained_variance.csv").exists():
        pca = pd.read_csv(workspace / "results/phase9/pca/phase9_pca_explained_variance.csv")
        lines += ["", f"First two PCA components retain {pca.Explained_Variance_Percent.iloc[:2].sum():.2f}% of variation."]
    lines += ["", "## Interpretation", "",
              "Decisions are algorithm classifications, not independently verified detection accuracy. "
              "The small, imbalanced dataset supports exploratory comparisons only. "
              "Autoencoder and Random Forest training results do not establish unseen-data performance. "
              "Wavelet energy is coefficient energy, not mechanical energy in joules.", "",
              "## Outputs", "", "- `results/`: stage tables, plots and reports",
              "- `logs/`: complete output from each stage", "- `manifest.json`: inputs, code, environment and stage status", ""]
    (workspace / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "raw_data")
    parser.add_argument("--output", type=Path, help="New directory; existing directories are never overwritten")
    parser.add_argument("--core-only", action="store_true", help="Run Phases 3–8 without exploratory ML")
    args = parser.parse_args(argv)
    manifest = None
    workspace = None
    try:
        required = ["numpy", "pandas", "scipy", "matplotlib", "pywt"]
        if not args.core_only:
            required += ["sklearn", "umap", "torch"]
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError(f"Missing dependencies: {', '.join(missing)}. Install requirements.txt first.")
        print("Checking dependency imports ...", flush=True)
        for name in required:
            try:
                importlib.import_module(name)
            except Exception as exc:
                raise ValueError(f"Cannot import {name}: {exc}. Reinstall requirements.txt in a fresh virtual environment.") from exc
        files, records = validate_inputs(args.input.resolve())
        workspace = (args.output or ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
        if workspace.exists():
            raise ValueError(f"Output directory already exists: {workspace}. Choose a new directory.")
        workspace.mkdir(parents=True)
        (workspace / "raw_data").mkdir()
        (workspace / "logs").mkdir()
        for path in files:
            shutil.copy2(path, workspace / "raw_data" / path.name)
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        (workspace / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
        manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "python": sys.version, "commit": git.stdout.strip(), "inputs": records,
                    "mode": "core" if args.core_only else "full", "stages": [],
                    "source_sha256": {str(p.relative_to(ROOT)): fingerprint(p) for p in sorted(ROOT.rglob("*.py"))
                                      if not any(part in {"runs", ".venv", "venv", ".git"} for part in p.relative_to(ROOT).parts)},
                    "settings": {"sampling_hz": 50, "baseline_samples": 100, "final_filter": "savitzky_golay", "vote_minimum": 2}}
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        for script, artifact in STAGES[:6] if args.core_only else STAGES:
            entry = {"script": script, "status": "running"}
            manifest["stages"].append(entry)
            (workspace / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Running {script} ...", flush=True)
            start = time.monotonic()
            log_path = workspace / "logs" / f"{Path(script).stem}.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, str(ROOT / script)], cwd=workspace, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            entry["seconds"] = round(time.monotonic() - start, 2)
            entry["returncode"] = result.returncode
            log_text = log_path.read_text(encoding="utf-8")
            if result.returncode or re.search(r"\[FAIL\]|\[CHECK[^\n]*\[FAIL\]|VALIDATION[^\n]*(?:FAILED|SOME CHECKS FAILED)", log_text):
                entry["status"] = "failed"
                raise RuntimeError(f"{script} failed. See {log_path}")
            check_stage(workspace, script, artifact, len(files))
            entry["status"] = "passed"
        manifest["status"] = "passed"
        write_report(workspace, manifest)
        manifest["artifacts"] = {str(path.relative_to(workspace)): fingerprint(path)
                                 for folder in ("results", "data")
                                 for path in sorted((workspace / folder).rglob("*")) if path.is_file()}
        print(f"Pipeline passed. Report: {workspace / 'REPORT.md'}", flush=True)
        return 0
    except Exception as exc:
        if manifest is not None:
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            if manifest["stages"] and manifest["stages"][-1]["status"] == "running":
                manifest["stages"][-1]["status"] = "failed"
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if manifest is not None:
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (workspace / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
