# Results and Discussion

## 1. Methodology
To rigorously identify the drivers of discrepancy between the InCore simulation and the "perfect" performance baseline, we employed a multi-stage feature selection and validation framework. This approach was designed to distinguish between statistical artifacts and genuine physical governing variables.

### 1.1 Data Preprocessing and Integrity
The target variable, `RPSS_vs_perfect`, represents the Ranked Probability Skill Score comparison. We identified a subset of records ($N \approx 12$) containing infinite values. Investigation revealed these cases corresponded to a `perfect_RPS` of 0.0, representing a mathematical singularity where the baseline assumes zero uncertainty. To preserve the statistical validity of the models, these records were treated as non-physical outliers and removed.

### 1.2 Feature Selection Framework
We utilized a "Filter-Wrapper" hybrid approach:
1.  **Mutual Information (Filter)**: We calculated the Mutual Information (MI) between all building features and the target. To establish a significance threshold, we introduced random "probe" features (Gaussian noise) into the dataset. Only features with an information gain exceeding this random baseline ($MI > MI_{random}$) were retained for further analysis.
2.  **Recursive Feature Elimination (Wrapper)**: On the subset of significant features, we applied Recursive Feature Elimination (RFE) using a Random Forest Regressor. This step iteratively removed the least important features to identify the minimal subset required for optimal prediction, accounting for feature redundancy and interaction effects.

### 1.3 Validation Strategy
To verify the robustness of the identified drivers, we performed:
*   **5-Fold Cross-Validation**: To ensure results were not overfit to a specific data split.
*   **Determinism Check**: We tested a minimal model using only the primary hazard and structural variables to assess if the outcome was deterministic.
*   **Circularity Investigation**: We rigorously tested the model's performance with and without the Hazard Intensity (`EF_scale`) to determine if its inclusion constituted data leakage or legitimate physical causation.

---

## 2. Results

### 2.1 Feature Significance
The Mutual Information analysis revealed that the `RPSS_vs_perfect` score is not random; **65 features** demonstrated statistical significance above the noise floor ($MI_{baseline} \approx 0.038$).
*   **Dominant Predictor**: The Hazard Intensity (`EF_scale`) yielded the highest information gain ($MI = 1.38$), significantly outperforming any individual building characteristic.
*   **Structural Predictors**: Among building features, the `Archetype` ($MI = 0.93$) and `Building Area` ($MI = 0.77$) were the strongest individual predictors.

### 2.2 Model Performance and Determinism
The RFE process converged on a subset of **20 features** that achieved a mean $R^2$ of **0.995** ($\pm 0.01$) in 5-fold cross-validation.
Further testing revealed that the system is effectively deterministic. A minimal model using **only** `EF_scale` and `Archetype` achieved an $R^2$ of **0.999**, indicating that the simulation's deviation from perfection is almost entirely a function of the wind speed and the structural system definition.

### 2.3 The Role of Hazard Intensity (Circularity vs. Causality)
A critical finding emerged from the "No-EF" investigation. When `EF_scale` was excluded from the model, predictive performance collapsed ($R^2$ dropped to $\approx 0.29$).
Furthermore, analysis of the target variable trend showed a clear systematic bias:
*   **Low Intensity (EF0-1)**: High error magnitude (Mean RPSS $\approx -2.5$ to $-17$).
*   **High Intensity (EF4)**: Near-perfect accuracy (Mean RPSS $\approx -0.07$).

---

## 3. Discussion and Physical Implications

### 3.1 Simulation Bias is Hazard-Dependent
The incredibly high correlation between `EF_scale` and the performance score is **not circularity**; it is a diagnosis of **systematic model bias**. The results demonstrate that the InCore simulation is well-calibrated for catastrophic events (EF4) but struggles significantly to represent variance or uncertainty in low-intensity scenarios. This implies that the fragility curves or damage states used in the simulation may be overly coarse or pessimistic at lower wind speeds, leading to large discrepancies with the "perfect" baseline.

### 3.2 The Primacy of Archetype
After hazard intensity, `Archetype` is the sole dominant building feature. This makes physical sense, as the archetype in fragility-based modeling encapsulates the entire structural system (load path, material, code vintage) into a single classification. Secondary geometric features like `Building Area` or `Number of Stories` are statistically significant primarily because they correlate with the archetype (e.g., larger buildings tend to be specific commercial archetypes), but they add negligible predictive power once the archetype is known.

### 3.3 Implications for Future Modeling
1.  **Calibration Focus**: Efforts to improve simulation accuracy should focus almost exclusively on **low-intensity calibration**. The model is already performing at a "perfect" level for high-intensity events.
2.  **Model Efficiency**: Future predictive models for this domain can be extremely parsimonious. There is no need to collect or model hundreds of granular building details (e.g., individual window attributes) to predict the reliability of the simulation; the `Archetype` classification alone captures the necessary structural physics.
