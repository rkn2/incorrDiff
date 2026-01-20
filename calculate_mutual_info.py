import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns

def main():
    # 1. Load Data
    input_file = 'ywang_quadstate_data_full_single_RPSS.csv'
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # 2. Preprocessing
    
    # Filter rows where target variable is available
    target_col = 'RPSS_vs_perfect'
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    print(f"Filtered {initial_len - len(df)} rows with missing '{target_col}'. Remaining: {len(df)}")

    # Columns to exclude (identifiers, post-event damage, uncertainty)
    # Using a broad exclusion list based on inspection
    exclude_keywords = [
        'source', 'town', 'NRHP', 'address', 'building_name', 'latitude', 'longitude', 
        'tornado', 'damage', 'unc', 'status_u', 'hazards_present_u'
    ]
    
    # Also exclude specific non-physics metadata
    drop_cols = {
        'complete_address', 'national_register_listing_year', 'source', 
        'building_urban_setting', 'building_position_on_street', # Maybe keep these? Let's keep for now but drop strictly location IDs
        'incore_RPS', 'perfect_RPS', target_col, 'building_count'
    }

    # Identify columns to drop based on keywords
    cols_to_drop = set(drop_cols)
    for col in df.columns:
        if any(keyword in col for keyword in exclude_keywords):
            cols_to_drop.add(col)
    
    # Be careful not to drop EF_scale if we want it as a control, though 'tornado' keyword might catch it.
    # The user asked to think critically about physics. 
    # EF_scale is the hazard intensity. It SHOULD be highly correlated. 
    # Let's keep EF_scale to see its dominance, but mark it.
    # Wait, 'tornado_EF_unc' is excluded by 'unc', 'tornado_start_lat' etc by 'tornado'. 
    # 'EF_scale' has 'EF_scale' name. Check keywords.
    # 'tornado' is in exclude_keywords. 'EF_scale' does not contain 'tornado'.
    
    print(f"Dropping {len(cols_to_drop)} columns containing keywords: {exclude_keywords} or specific drop list.")
    
    feature_df = df.drop(columns=list(cols_to_drop))
    
    # Handle object columns (categorical)
    # Fill missing values with 'Missing' for object columns, and median for numeric
    # Actually, for MI with mixed types, we need to be careful.
    # Simple strategy: Ordinal Encode everything that is object.
    
    X = feature_df.copy()
    y = df[target_col]

    # Clean target variable: remove infinite values
    y = y.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows where y is NaN again (in case inf became NaN)
    valid_indices = y.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    # Impute missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Missing')
        else:
            X[col] = X[col].fillna(X[col].median())

    # Add Random Probe Features (Baselines)
    np.random.seed(42)
    X['random_continuous'] = np.random.uniform(0, 100, size=len(X))
    X['random_discrete'] = np.random.randint(0, 5, size=len(X))

    # Encode categorical variables
    encoder = OrdinalEncoder()
    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X[cat_cols] = encoder.fit_transform(X[cat_cols])
    
    # Identify discrete features (integer encoded) for MI
    discrete_mask = []
    for col in X.columns:
        # If it was object or has < 20 unique values (like number of stories) OR is our random discrete
        if col in cat_cols or X[col].nunique() < 20 or col == 'random_discrete': 
            discrete_mask.append(True)
        else:
            discrete_mask.append(False)

    print("Calculating Mutual Information scores...")
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)
    
    mi_series = pd.Series(mi_scores, index=X.columns)
    mi_series = mi_series.sort_values(ascending=False)
    
    # Determine Baseline Threshold (Max of random probes)
    random_baseline = mi_series[['random_continuous', 'random_discrete']].max()
    print(f"\nRandom Baseline MI Score: {random_baseline:.4f}")

    # Count significant features
    significant_features = mi_series[mi_series > random_baseline]
    num_sig = len(significant_features)
    print(f"Number of features above random baseline: {num_sig}")
    
    # Decide how many to plot: All significant + 5 context, capped at 50
    top_n = min(num_sig + 5, 50)
    
    print(f"\nTop {top_n} Features by Mutual Information:")
    print(mi_series.head(top_n))

    # 3. Visualization
    plt.figure(figsize=(10, 12))
    # Color bar based on threshold
    data_to_plot = mi_series.head(top_n)
    colors = ['red' if x <= random_baseline else 'skyblue' for x in data_to_plot.values]
    
    sns.barplot(x=data_to_plot.values, y=data_to_plot.index, palette=colors)
    plt.axvline(x=random_baseline, color='red', linestyle='--', label=f'Random Baseline ({random_baseline:.3f})')
    
    plt.title(f'Top {top_n} Features vs {target_col} (Mutual Information)\n({num_sig} features > random noise)')
    plt.xlabel('Mutual Information Score')
    plt.legend()
    plt.tight_layout()
    
    output_plot = 'mutual_info_scores.png'
    plt.savefig(output_plot, dpi=300)
    print(f"\nPlot saved to {output_plot}")

if __name__ == "__main__":
    main()
