import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def main():
    # 1. Load same cleaned data logic (simplified for brevity)
    input_file = 'ywang_quadstate_data_full_single_RPSS.csv'
    print(f"Loading data from {input_file} for RFE...")
    df = pd.read_csv(input_file)
    target_col = 'RPSS_vs_perfect'
    
    # Preprocessing (same as before)
    df = df.dropna(subset=[target_col])
    # Clean infs
    y = df[target_col]
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_indices = y.dropna().index
    df = df.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # Drop columns
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

    # 2. Setup RFE
    # Use RandomForest as the estimator (since user uses tree models)
    forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # RFE: Select top 20 features (arbitrary, but good for comparison with MI top 20)
    # Recursively removes weakest features based on feature_importances_
    selector = RFE(estimator=forest, n_features_to_select=20, step=1)
    
    print("Running Recursive Feature Elimination (this may take a moment)...")
    selector.fit(X, y)
    
    # Get selected features
    selected_mask = selector.support_
    selected_features = X.columns[selected_mask]
    ranking = selector.ranking_
    
    print("\nRFE Selected Top 20 Features:")
    print(selected_features.tolist())
    
    # Train model on these 20 to check correlation/performance
    X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
    forest.fit(X_train, y_train)
    score = r2_score(y_test, forest.predict(X_test))
    print(f"\nRandom Forest R2 Score with top 20 RFE features: {score:.4f}")

if __name__ == "__main__":
    main()
