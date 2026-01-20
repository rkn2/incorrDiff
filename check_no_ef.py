import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, KFold

def main():
    print("--- Circularity Check: Removing 'EF_scale' ---")
    
    # 1. Load Data (Simplified Copy-Paste Logic)
    input_file = 'ywang_quadstate_data_full_single_RPSS.csv'
    df = pd.read_csv(input_file)
    target_col = 'RPSS_vs_perfect'
    
    # Preprocessing
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_indices = y.dropna().index
    df = df.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # Drops
    exclude_keywords = [
        'source', 'town', 'NRHP', 'address', 'building_name', 'latitude', 'longitude', 
        'tornado', 'damage', 'unc', 'status_u', 'hazards_present_u'
    ]
    drop_cols = {
        'complete_address', 'national_register_listing_year', 'source', 
        'building_urban_setting', 'building_position_on_street',
        'incore_RPS', 'perfect_RPS', target_col, 'building_count', 
        'EF_scale' # <--- EXPLICITLY REMOVING EF SCALE
    }
    cols_to_drop = set(drop_cols)
    for col in df.columns:
        if any(keyword in col for keyword in exclude_keywords):
            cols_to_drop.add(col)
            
    X = df.drop(columns=list(cols_to_drop))
    print(f"Features Remaining: {X.shape[1]} (EF_scale extracted? {'EF_scale' not in X.columns})")
    
    # Impute/Encode
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Missing')
        else:
            X[col] = X[col].fillna(X[col].median())

    encoder = OrdinalEncoder()
    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X[cat_cols] = encoder.fit_transform(X[cat_cols])
        
    # 2. Check MI without EF
    print("\nCalculating Mutual Information (No EF)...")
    # Quick discrete mask estimate
    discrete_mask = [True if (c in cat_cols or X[c].nunique() < 20) else False for c in X.columns]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    print("Top 10 Features (No EF):")
    print(mi_series.head(10))
    
    # 3. Validation Model (Random Forest)
    print("\nValidating Model Performance (5-Fold CV)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    scores = cross_val_score(forest, X, y, cv=kf, scoring='r2')
    print(f"R2 Scores (No EF): {scores}")
    print(f"Mean R2: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

if __name__ == "__main__":
    main()
