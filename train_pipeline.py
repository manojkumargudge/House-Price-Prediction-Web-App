# src/train_pipeline.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Paths
DATA_PATH = os.path.join("..", "data", "train.csv")  # dataset path
SAVE_PATH = os.path.join("..", "house_price_pipeline.pkl")  # where model will be saved

# Chosen numeric features
numeric_features = [
    "OverallQual", "GrLivArea", "GarageCars",
    "GarageArea", "TotalBsmtSF", "FullBath", "YearBuilt"
]
target = "SalePrice"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=[target])
    return df

def build_pipeline():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features)
    ], remainder="drop")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline

def main():
    print("Loading data...")
    df = load_data()
    X = df[numeric_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    preds = pipeline.predict(X_test)
    mse = np.mean((preds - y_test) ** 2)
    r2 = pipeline.score(X_test, y_test)
    print(f"MSE: {mse:.2f}, R2: {r2:.4f}")

    # Save model
    joblib.dump(pipeline, SAVE_PATH)
    print(f"✅ Model saved as {SAVE_PATH}")

if __name__ == "__main__":
    main()
