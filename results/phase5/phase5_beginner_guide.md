# Beginner's Guide to Phase 5: Engineering Signal Characterization

Welcome! If you are new to Fiber Bragg Grating (FBG) sensor analysis or signal processing, this guide will explain **what Phase 5 does, why we built it, and what all the calculated numbers mean in simple terms.**

---

## 1. The Big Picture: What is Phase 5?

Imagine an impact (like a small hammer strike) hits a metal structure equipped with fiber optic sensors. The sensors detect vibrations and strain waves, producing a "wiggle" in the light wavelength.

Before feeding these raw wiggles into a complex AI/Machine Learning model, **good engineering requires understanding the physical signal first.** 

> 💡 **Doctor Analogy**: Before a doctor runs a complex AI diagnostic, they first measure your basic vitals: body temperature, blood pressure, and heart rate. **Phase 5 measures the "vitals" of the impact signal.**

Phase 5 takes the detected impact events from Phase 4 and calculates **13 core engineering features** for every sensor.

---

## 2. The 3 Sensors and Material Coatings

In this experiment, we have **3 FBG sensor channels**, each protected by a different material packaging:

| Sensor Channel | Material Packaging | Description |
| :--- | :--- | :--- |
| **`FBG2`** | **Bare** | Naked silica glass fiber with **no protective metal layer**. Direct contact with structural strain. |
| **`FBG1`** | **Copper** | Fiber wrapped in **soft, ductile copper metal**. Acts as a compliant protective layer. |
| **`FBG3`** | **Steel** | Fiber encapsulated in **hard, stiff stainless steel**. Provides maximum mechanical protection. |

---

## 3. Simple Guide to the 13 Engineering Features

Here is what each of the 13 features means in everyday language, along with its physical units:

### 1. Peak Shift (Unit: `nm` - nanometers)
* **What it means**: The maximum distance the light wavelength jumped away from its rest position when the impact hit.
* **Analogy**: How high a person jumped off the ground after getting startled.
* **Why it matters**: Tells us the maximum localized strain/stretching felt by the fiber.

### 2. Residual Shift (Unit: `nm` - nanometers)
* **What it means**: The remaining wavelength shift left over *after* the impact vibration has stopped.
* **Analogy**: If you stretch a rubber band too hard, it stays slightly loose. That leftover stretch is the residual shift.
* **Why it matters**: Tells us if the impact caused permanent deformation or lingering stress in the material.

### 3. Rise Time (Unit: `s` - seconds)
* **What it means**: How much time it took for the signal to shoot up from 10% to 90% of its peak value.
* **Analogy**: How quickly a sports car accelerates from 10 mph to 90 mph.
* **Why it matters**: Fast rise time means a sharp, sudden impact wave; slow rise time means a soft or cushioned impact.

### 4. Recovery Time (Unit: `s` - seconds)
* **What it means**: How long it took for the signal to stop vibrating and return back to normal baseline after reaching its peak.
* **Analogy**: How long a plucked guitar string continues to vibrate before going completely silent.
* **Why it matters**: Measures how quickly energy dissipates and dies out.

### 5. Peak Width / FWHM (Unit: `s` - seconds)
* **What it means**: Full Width at Half Maximum — the duration (width in seconds) of the main impact spike measured at 50% of its height.
* **Analogy**: The width of a mountain at half its peak elevation.
* **Why it matters**: Broad peaks indicate prolonged impact contact; narrow peaks indicate quick impulse hits.

### 6. Maximum Slope (Unit: `nm/s` - nanometers per second)
* **What it means**: The steepest rate of signal change per second $\max(|d\text{Signal}/dt|)$.
* **Analogy**: The steepest section of a roller coaster drop.
* **Why it matters**: Measures the maximum speed of stress wave arrival.

### 7. RMS - Root Mean Square (Unit: `nm` - nanometers)
* **What it means**: The overall effective average power/magnitude of the signal wiggle.
* **Analogy**: The overall volume level of a music track (combining loud peaks and quiet tails).
* **Why it matters**: Gives a single number for how "intense" the total vibration event was.

### 8. Signal Energy (Unit: `nm²·s` - nanometer squared seconds)
* **What it means**: Integrated total mathematical energy contained in the impact wave over time ($\int x(t)^2 dt$).
* **Analogy**: Total electrical energy consumed by a lightbulb while it was turned on.
* **Why it matters**: Combines both the strength of the impact and how long it lasted.

### 9. Peak-to-Peak (Unit: `nm` - nanometers)
* **What it means**: The total vertical distance between the highest crest and the lowest trough of the signal.
* **Analogy**: The difference between the highest tide and lowest tide of the ocean in a day.
* **Why it matters**: Shows the full dynamic swing range of the sensor.

### 10. Variance (Unit: `nm²` - nanometer squared)
* **What it means**: How widely scattered or spread out the signal values are around their average baseline.
* **Analogy**: Weather variance — whether temperatures stay steady or jump wildly between hot and cold.
* **Why it matters**: Measures overall signal instability during the event.

### 11. Standard Deviation (Unit: `nm` - nanometers)
* **What it means**: The square root of variance $\sqrt{\text{Variance}}$ — average amount signal values deviate from baseline.
* **Analogy**: Standard margin of error in a poll.
* **Why it matters**: Measures typical magnitude of vibration fluctuation.

### 12. Distributional Entropy (Unit: `bits`)
* **What it means**: A measure of signal randomness, complexity, and unpredictability based on Shannon Information Theory.
* **Analogy**: White noise on a TV has high entropy (random); a pure smooth tuning fork chime has low entropy (predictable).
* **Why it matters**: Complex metallic reflections and ringing create higher entropy.

### 13. Area Under Curve - AUC (Unit: `nm·s` - nanometer seconds)
* **What it means**: The cumulative area under the baseline-corrected signal curve ($\int |x(t)| dt$).
* **Analogy**: Total distance traveled on a road trip.
* **Why it matters**: Quantifies the total cumulative strain impulse over time.

---

## 4. Why Does Bare > Copper > Steel? (The Physics Explained Simply)

Our calculated results revealed a clear ranking in Peak Shift:

$$\text{Bare (FBG2: 0.0298 nm)} > \text{Copper (FBG1: 0.0078 nm)} > \text{Steel (FBG3: 0.0028 nm)}$$

Why does this happen?

```text
Bare FBG (FBG2)
  ↓
No protective metal shell
  ↓
Direct strain transfer from structural surface into glass core
  ↓
HIGHEST Peak Shift (0.0298 nm)

Copper Packaging (FBG1)
  ↓
Wrapped in soft, ductile copper metal
  ↓
Copper metal layer stretches and absorbs part of the impact force
  ↓
MEDIUM Peak Shift (0.0078 nm)

Steel Packaging (FBG3)
  ↓
Wrapped in hard, stiff stainless steel
  ↓
High stiffness resists deformation and carries structural load around the sensor
  ↓
LOWEST Peak Shift (0.0028 nm)
```

---

## 5. How the Code Files Work Together

The Phase 5 code is organized cleanly inside `src/`:

```text
FBG_Impact_Analysis/
├── src/
│   ├── engineering_features.py   # Pure math functions for all 13 features
│   ├── feature_extraction.py     # Reads data files & applies math functions
│   └── comparison_analysis.py    # Compares Bare vs Copper vs Steel & generates plots
├── phase5.py                     # Main script you run from terminal
└── results/phase5/               # Output folder containing CSVs, Markdown reports & plots
```

---

## 6. How to Run Phase 5

To run the complete Phase 5 pipeline yourself:

1. Open your terminal in the repository root directory.
2. Run the command:
   ```bash
   python phase5.py
   ```

3. Output files will instantly appear inside `results/phase5/`:
   - `phase5_all_features.csv`: Full detailed feature table for all datasets.
   - `phase5_material_comparison.csv`: Bare vs Copper vs Steel comparison table.
   - `phase5_engineering_explanation.md`: Detailed physics explanations.
   - `phase5_summary.md`: Summary report.
   - `results/phase5/plots/`: 13 generated bar chart images comparing all 3 materials for every feature!
