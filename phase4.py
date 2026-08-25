import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from peak_detection import detect_peak
from threshold_detection import detect_threshold
from derivative_detection import detect_derivative
from changepoint_detection import detect_changepoint


# ============================================================
# PHASE 4 - IMPACT DETECTION
# ============================================================

def main():

    # --------------------------------------------------------
    # INPUT / OUTPUT DIRECTORIES
    # --------------------------------------------------------

    # Phase 3 verified output
    data_dir = "data/processed/final_phase_input"

    # Only final Phase 4 results will be stored here
    results_dir = "results/phase4"

    os.makedirs(results_dir, exist_ok=True)

    # --------------------------------------------------------
    # FIND PHASE 3 OUTPUT FILES
    # --------------------------------------------------------

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        print("ERROR: No Phase 3 input CSV files found.")
        print(f"Expected files inside: {data_dir}")
        return

    # Sort datasets by expert number
    def extract_expert_num(filename):
        base = os.path.basename(filename)

        try:
            return int(base.split("expert")[1].split(".")[0])
        except (IndexError, ValueError):
            return 0

    csv_files.sort(key=extract_expert_num)

    print("=" * 70)
    print("PHASE 4 - IMPACT DETECTION")
    print("=" * 70)
    print(f"Input directory : {data_dir}")
    print(f"Output directory: {results_dir}")
    print(f"Datasets found  : {len(csv_files)}")
    print("=" * 70)

    # --------------------------------------------------------
    # VARIABLES
    # --------------------------------------------------------

    all_results = []
    summary_data = []

    fbgs = ["FBG1", "FBG2", "FBG3"]

    total_impacts = 0
    total_no_impacts = 0

    fbg_counts = {
        "FBG1": {"impact": 0, "no_impact": 0},
        "FBG2": {"impact": 0, "no_impact": 0},
        "FBG3": {"impact": 0, "no_impact": 0}
    }

    # --------------------------------------------------------
    # PROCESS EACH DATASET
    # --------------------------------------------------------

    for file_path in csv_files:

        filename = os.path.basename(file_path)
        expert_num = extract_expert_num(filename)
        expert_name = f"Expert {expert_num}"

        print("\n" + "=" * 70)
        print(f"PROCESSING: {expert_name}")
        print(f"File: {filename}")
        print("=" * 70)

        # Load Phase 3 output
        df = pd.read_csv(file_path)

        # Verify required columns
        required_columns = [
            "Time",
            "FBG1_processed",
            "FBG2_processed",
            "FBG3_processed"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            print(f"ERROR: Missing columns: {missing_columns}")
            continue

        time_series = df["Time"]

        summary_row = {
            "Dataset": expert_name
        }

        # ----------------------------------------------------
        # PROCESS FBG1, FBG2, FBG3
        # ----------------------------------------------------

        for fbg in fbgs:

            col_name = f"{fbg}_processed"
            signal_series = df[col_name]

            print(f"\n--- {fbg} ---")

            # ------------------------------------------------
            # RUN FOUR DETECTION METHODS
            # ------------------------------------------------

            peak_res, peak_time, p_mean, p_noise, p_thresh, p_val, p_ratio = (
                detect_peak(time_series, signal_series)
            )

            thresh_res, thresh_time, t_mean, t_noise, t_thresh, t_val, t_ratio = (
                detect_threshold(time_series, signal_series)
            )

            deriv_res, deriv_time, d_mean, d_noise, d_thresh, d_val, d_ratio = (
                detect_derivative(time_series, signal_series)
            )

            cp_res, cp_time, c_mean, c_noise, c_thresh, c_val, c_ratio = (
                detect_changepoint(time_series, signal_series)
            )

            # ------------------------------------------------
            # ENSEMBLE DECISION
            # ------------------------------------------------

            methods_detected = sum([
                peak_res,
                thresh_res,
                deriv_res,
                cp_res
            ])

            methods_not_detected = 4 - methods_detected

            # Current ensemble rule:
            # 2 or more of 4 methods => IMPACT
            final_result = (
                "IMPACT"
                if methods_detected >= 2
                else "NO IMPACT"
            )

            # ------------------------------------------------
            # CONSOLE OUTPUT
            # ------------------------------------------------

            print(
                f"Peak        : "
                f"{'IMPACT' if peak_res else 'NO IMPACT'}"
                f" | Time: {peak_time if peak_time is not None else 'N/A'}"
            )

            print(
                f"Threshold   : "
                f"{'IMPACT' if thresh_res else 'NO IMPACT'}"
                f" | Time: {thresh_time if thresh_time is not None else 'N/A'}"
            )

            print(
                f"Derivative  : "
                f"{'IMPACT' if deriv_res else 'NO IMPACT'}"
                f" | Time: {deriv_time if deriv_time is not None else 'N/A'}"
            )

            print(
                f"ChangePoint : "
                f"{'IMPACT' if cp_res else 'NO IMPACT'}"
                f" | Time: {cp_time if cp_time is not None else 'N/A'}"
            )

            print(f"Methods detecting impact: {methods_detected}/4")
            print(f"FINAL RESULT: {final_result}")

            # ------------------------------------------------
            # STORE METHOD RESULTS
            # ------------------------------------------------

            all_results.append({
                "Dataset": expert_name,
                "FBG": fbg,
                "Peak": "YES" if peak_res else "NO",
                "Threshold": "YES" if thresh_res else "NO",
                "Derivative": "YES" if deriv_res else "NO",
                "ChangePoint": "YES" if cp_res else "NO",
                "Methods_Detected": methods_detected,
                "Methods_Not_Detected": methods_not_detected,
                "Final_Result": final_result
            })

            # ------------------------------------------------
            # FINAL RESULT PLOT
            # ------------------------------------------------
            #
            # This is a RESULT plot, not a diagnostic plot.
            #
            # It shows:
            #   - filtered FBG signal
            #   - detected impact points
            #   - final impact/no-impact decision
            #
            # ------------------------------------------------

            plt.figure(figsize=(12, 6))

            plt.plot(
                time_series,
                signal_series,
                label=f"{fbg} Filtered Signal"
            )

            # Mark detections from the four methods
            if peak_res and peak_time is not None:
                plt.axvline(
                    peak_time,
                    linestyle="--",
                    label="Peak Detection"
                )

            if thresh_res and thresh_time is not None:
                plt.axvline(
                    thresh_time,
                    linestyle=":",
                    label="Threshold Detection"
                )

            if deriv_res and deriv_time is not None:
                plt.axvline(
                    deriv_time,
                    linestyle="-.",
                    label="Derivative Detection"
                )

            if cp_res and cp_time is not None:
                plt.axvline(
                    cp_time,
                    linestyle="--",
                    label="Change-Point Detection"
                )

            plt.title(
                f"{expert_name} - {fbg} - {final_result}"
            )

            plt.xlabel("Time (s)")
            plt.ylabel("Processed Signal")

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_filename = (
                f"{expert_name.replace(' ', '')}_{fbg}_Impact_Result.png"
            )

            plt.savefig(
                os.path.join(results_dir, plot_filename),
                dpi=300
            )

            plt.close()

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            summary_row[f"{fbg}_Result"] = final_result
            summary_row[f"{fbg}_Method_Count"] = methods_detected

            if final_result == "IMPACT":
                total_impacts += 1
                fbg_counts[fbg]["impact"] += 1
            else:
                total_no_impacts += 1
                fbg_counts[fbg]["no_impact"] += 1

        summary_data.append(summary_row)

    # ========================================================
    # SAVE FINAL METHOD RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        os.path.join(
            results_dir,
            "phase4_method_results.csv"
        ),
        index=False
    )

    # ========================================================
    # SAVE FINAL SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(summary_data)

    summary_columns = [
        "Dataset",
        "FBG1_Result",
        "FBG2_Result",
        "FBG3_Result",
        "FBG1_Method_Count",
        "FBG2_Method_Count",
        "FBG3_Method_Count"
    ]

    summary_df = summary_df[summary_columns]

    summary_df.to_csv(
        os.path.join(
            results_dir,
            "phase4_summary.csv"
        ),
        index=False
    )

    # ========================================================
    # SAVE FINAL REPORT
    # ========================================================

    with open(
        os.path.join(results_dir, "phase4_report.txt"),
        "w"
    ) as f:

        f.write("PHASE 4 - IMPACT DETECTION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total datasets processed = {len(csv_files)}\n")
        f.write(f"Total FBG analyses = {len(all_results)}\n")
        f.write(f"Total impacts = {total_impacts}\n")
        f.write(f"Total no-impact cases = {total_no_impacts}\n\n")

        f.write("BREAKDOWN BY FBG\n")
        f.write("-" * 30 + "\n")

        for fbg in fbgs:

            f.write(f"\n{fbg}:\n")
            f.write(
                f"  Impacts: "
                f"{fbg_counts[fbg]['impact']}\n"
            )
            f.write(
                f"  No Impacts: "
                f"{fbg_counts[fbg]['no_impact']}\n"
            )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)

    print(f"Datasets processed : {len(csv_files)}")
    print(f"FBG analyses       : {len(all_results)}")
    print(f"Total IMPACT       : {total_impacts}")
    print(f"Total NO IMPACT    : {total_no_impacts}")

    print("\nResults saved to:")
    print(results_dir)

    print("\nGenerated:")
    print("  - phase4_method_results.csv")
    print("  - phase4_summary.csv")
    print("  - phase4_report.txt")
    print("  - Impact result plots")

    print("=" * 70)


if __name__ == "__main__":
    main()