import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score

def load_and_prep_data():
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
    
    # Clean Features
    exclude_keywords = [
        'source', 'town', 'NRHP', 'address', 'building_name', 'latitude', 'longitude', 
        'tornado', 'damage', 'unc', 'status_u', 'hazards_present_u'
    ]
    drop_cols = {
        'complete_address', 'national_register_listing_year', 'source', 
        'building_urban_setting', 'building_position_on_street',
        'incore_RPS', 'perfect_RPS', target_col, 'building_count'
    }
    cols_to_drop = set(drop_cols)
    for col in df.columns:
        if any(keyword in col for keyword in exclude_keywords):
            cols_to_drop.add(col)
            
    X = df.drop(columns=list(cols_to_drop))
    
    # Impute and Encode
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Missing')
        else:
            X[col] = X[col].fillna(X[col].median())

    encoder = OrdinalEncoder()
    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X[cat_cols] = encoder.fit_transform(X[cat_cols])
        
    return X, y

def main():
    X, y = load_and_prep_data()
    print(f"Data Loaded: {X.shape[0]} rows, {X.shape[1]} features")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Test 1: Full RFE Subset (Top 20 from previous run)
    # Re-running RFE internally to simulate the pipeline or just using the known top features?
    # Let's run a quick RFE to get the features again to be fair
    print("\nRunning RFE to select Top 20 features...")
    selector = RFE(estimator=forest, n_features_to_select=20, step=1)
    selector.fit(X, y)
    top_20 = X.columns[selector.support_]
    print(f"Features: {top_20.tolist()}")
    
    print("\n--- Validation 1: Top 20 Features (5-Fold CV) ---")
    cv_scores = cross_val_score(forest, X[top_20], y, cv=kf, scoring='r2')
    print(f"R2 Scores: {cv_scores}")
    print(f"Mean R2: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Test 2: "Determinism Check" (Archetype + EF_Scale only)
    # Hypothesis: Does the target just depend on these two?
    # "EF_scale" and "archetype"
    minimal_feats = ['EF_scale', 'archetype']
    print(f"\n--- Validation 2: Determinism Check {minimal_feats} ---")
    
    if all(f in X.columns for f in minimal_feats):
        cv_scores_min = cross_val_score(forest, X[minimal_feats], y, cv=kf, scoring='r2')
        print(f"R2 Scores: {cv_scores_min}")
        print(f"Mean R2: {cv_scores_min.mean():.4f} (+/- {cv_scores_min.std()*2:.4f})")
    else:
        print(f"Could not run check, features missing: {[f for f in minimal_feats if f not in X.columns]}")

    # Test 3: Random Baseline
    # Permutation test logic: Shuffle Y and predict
    print("\n--- Validation 3: Permutation Test (Random Shuffle) ---")
    y_shuffled = y.sample(frac=1, random_state=42).reset_index(drop=True)
    # Scikit-learn cross_val_score expects aligned X,y. If we shuffle y only...
    # Just manual loop
    
    perm_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index][top_20], X.iloc[test_index][top_20]
        y_train = y_shuffled.iloc[train_index] # Trained on noise
        y_test = y_shuffled.iloc[test_index]   # Test on noise
        
        forest.fit(X_train, y_train)
        perm_scores.append(r2_score(y_test, forest.predict(X_test)))
    
    print(f"Permuted R2 Scores: {perm_scores}")
    print(f"Mean Permuted R2: {np.mean(perm_scores):.4f}")

if __name__ == "__main__":
    main()
