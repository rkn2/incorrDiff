# InCore vs Perfect Performance Analysis (incorrDiff)

This repository analyzes the drivers of simulation error in InCore models by comparing their "Ranked Probability Skill Score" (RPSS) against a perfect baseline.

## Key Scrips

1.  `calculate_mutual_info.py`:
    *   **Method**: Filter Method (Mutual Information).
    *   **Features**: Includes a random probe baseline to determine statistical significance.
    *   **Output**: `mutual_info_scores.png` (Visualization of top features).

2.  `run_rfe.py`:
    *   **Method**: Wrapper Method (Recursive Feature Elimination with Random Forest).
    *   **Purpose**: Optimizes the feature set for predictive performance, removing redundancy.

3.  `validate_robustness.py`:
    *   **Method**: 5-Fold Cross Validation.
    *   **Purpose**: Verifies the $R^2$ scores and performs a "Determinism Check" (Archetype + Hazard only).

4.  `check_no_ef.py`:
    *   **Purpose**: Investigates the impact of excluding Hazard Intensity (`EF_scale`) to test for circularity.

## Documentation
*   [Analysis Report & Methodology](analysis_report.md): Detailed explanation of the choices made regarding feature selection, infinite value handling, and the inclusion of hazard intensity variables.

## Usage
Run the main analysis script:
```bash
python3 calculate_mutual_info.py
```
Run the validation suite:
```bash
python3 validate_robustness.py
```
