# Analysis Report: InCore Performance Score Drivers

## Overview
This repository contains the code and analysis results identifying the primary drivers of the "InCore vs Perfect" performance score (`RPSS_vs_perfect`). The goal was to determine which building features and event characteristics explain the discrepancy between the simulation and the "perfect" baseline.

## Methodology & Critical Decisions

### 1. Handling of Target Variable (`RPSS_vs_perfect`)
*   **Infinite Values**: We identified ~12 records where the target score was infinite (likely due to a `perfect_RPS` of 0.0). These were treated as physical impossibilities/divide-by-zero errors and **excluded** from the analysis to prevent statistical corruption.
*   **Directionality**: We verified that higher `EF_scale` values (3/4) correlate with `RPSS` scores closer to zero (better accuracy), while low EF values show massive error. This confirmed a systematic bias in the simulation's calibration for low-intensity events.

### 2. Feature Selection Strategy
We employed a two-stage approach to ensure both physical discovery and model efficiency:

#### Stage A: Mutual Information (Filter Method)
*   **Purpose**: To detect *any* statistical dependency between features and the target, regardless of model structure.
*   **Baseline**: We introduced random probe features (noise) to establish a significance floor ($MI \approx 0.038$).
*   **Finding**: **65 features** proved to be statistically significant (better than random noise), providing a comprehensive "Physics Discovery" list for researchers.

#### Stage B: Recursive Feature Elimination (Wrapper Method)
*   **Purpose**: To optimize for predictive power and identify "Team Player" variables that only working in combination.
*   **Result**: Reduced the 65 candidate features to a subset of **20 features** that maximized model performance ($R^2 \approx 0.999$).

### 3. The "EF_Scale" Dilemma: Circularity vs. Causality
A critical investigation was performed to decide whether to include `EF_scale` (Wind Speed/Hazard Intensity) as a predictor.
*   **Concern**: Is using wind to predict damage circular?
*   **Investigation**: We ran the analysis **without** `EF_scale`.
*   **Outcome**: The model performance collapsed ($R^2$ dropped from $0.99$ to $\sim 0.29$).
*   **Conclusion**: The hazard intensity is not just a correlate; it is the **fundamental driver** of the performance score. Excluding it removes the "load" from the "load-resistance" physics equation.
*   **Decision**: `EF_scale` was RETAINED. The analysis correctly identifies that **Simulated Error is primarily a function of Hazard Intensity.**

---

## Key Findings
1.  **Deterministic Nature**: The `RPSS_vs_perfect` score is effectively a deterministic function of **Hazard Intensity** (`EF_scale`) and **Building Archetype**. A model using only these two inputs achieves $R^2 \approx 0.999$.
2.  **Simulation Bias**: The simulation is highly accurate for major events (EF4) but systematically inaccurate for minor events (EF0/1), suggesting a need for recalibration of the fragility curves at low intensities.
3.  **Physical Features**: While 65 features are statistically significant, features like `building_area`, `foundation_type`, and `occupancy` are the strongest secondary indicators after the primary archetype definition.
