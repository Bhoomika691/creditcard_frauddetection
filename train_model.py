from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================
BASE = Path(__file__).resolve().parent

DATA_PATH = BASE / "data" / "indian_credit_card_fraud_multiclass_100k.csv"


MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ============================================================
# 2. FEATURES & TARGET
# ============================================================
FEATURES = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "transaction_type",
    "location",
    "transactions_last_24h",
    "avg_transaction_amount",
    "international",
    "distance_from_home_km"
]

TARGET = "is_fraud"

RAW_NUMERIC_FEATURES = [
    "amount",
    "transaction_hour",
    "transactions_last_24h",
    "avg_transaction_amount",
    "distance_from_home_km"
]

ENGINEERED_FEATURES = [
    "amount_ratio",
    "amount_diff",
    "is_late_night",
    "exceeds_std_limit",
    "high_value_tx",
    "limit_utilization_ratio"
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES

CATEGORICAL_FEATURES = [
    "merchant_category",
    "transaction_type",
    "location",
    "international"
]


# ============================================================
# 3. LOAD DATA
# ============================================================
print(f"\n1. Loading dataset from: {DATA_PATH.name}...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Rows:", len(df))
print("Columns:", len(df.columns))

required_columns = FEATURES + [TARGET]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError("Missing columns: " + ", ".join(missing_columns))

df = df[required_columns].copy()


# ============================================================
# 4. CLEAN DATA & FEATURE ENGINEERING
# ============================================================
print("\n2. Cleaning data and engineering features...")

df = df.drop_duplicates()

for col in CATEGORICAL_FEATURES:
    df[col] = df[col].astype(str).str.strip().str.title()
    df[col] = df[col].replace({"Nan": np.nan, "None": np.nan, "": np.nan, "N/A": np.nan})

def normalize_international(val):
    if pd.isna(val):
        return "No"
    val_str = str(val).strip().lower()
    if val_str in ["yes", "y", "true", "1", "1.0", "international", "intl"]:
        return "Yes"
    if val_str in ["no", "n", "false", "0", "0.0", "domestic"]:
        return "No"
    return "No"

df["international"] = df["international"].map(normalize_international)

for column in RAW_NUMERIC_FEATURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=[TARGET])

# Map multi-class target labels to integers: Low=0, Medium=1, High=2
target_mapping = {"Low": 0, "Medium": 1, "High": 2, 0: 0, 1: 1, 2: 2, "0": 0, "1": 1, "2": 2}
df[TARGET] = df[TARGET].astype(str).str.strip().str.title().map(target_mapping)
df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)

df.loc[df["amount"] <= 0, "amount"] = np.nan
df.loc[~df["transaction_hour"].between(0, 23), "transaction_hour"] = np.nan
df.loc[df["transactions_last_24h"] < 0, "transactions_last_24h"] = np.nan
df.loc[df["avg_transaction_amount"] <= 0, "avg_transaction_amount"] = np.nan
df.loc[df["distance_from_home_km"] < 0, "distance_from_home_km"] = np.nan

# Feature Engineering
df["amount_ratio"] = df["amount"] / (df["avg_transaction_amount"] + 1.0)
df["amount_diff"] = df["amount"] - df["avg_transaction_amount"]
df["is_late_night"] = df["transaction_hour"].between(0, 4).astype(float)
df["exceeds_std_limit"] = (df["amount"] > 200000).astype(float)
df["high_value_tx"] = (df["amount"] >= 50000).astype(float)
df["limit_utilization_ratio"] = df["amount"] / 200000.0

print("Rows after cleaning:", len(df))


# ============================================================
# 5. BASIC EDA
# ============================================================
print("\n3. Basic EDA")
print("\nRisk Class distribution:")
print(
    df[TARGET]
    .value_counts()
    .rename(index={0: "Low", 1: "Medium", 2: "High"})
)


# ============================================================
# 6. SEPARATE INPUT AND OUTPUT
# ============================================================
print("\n4. Preparing X and y...")
ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = df[ALL_MODEL_FEATURES]
y = df[TARGET]

print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================
print("\n5. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 8. PREPROCESSING PIPELINE
# ============================================================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])


# ============================================================
# 9. CREATE CANDIDATE MODELS
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=4,
        objective="multi:softprob",
        num_class=3
    )
}


# ============================================================
# 10. CROSS-VALIDATION
# ============================================================
print("\n6. Comparing models...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

for name, model in models.items():
    print("\nModel:", name)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_weighted",
            "recall_weighted"
        ],
        n_jobs=2
    )

    results[name] = {
        "accuracy": round(float(scores["test_accuracy"].mean()), 4),
        "precision": round(float(scores["test_precision_weighted"].mean()), 4),
        "recall": round(float(scores["test_recall_weighted"].mean()), 4),
        "f1": round(float(scores["test_f1_weighted"].mean()), 4),
        "f1_macro": round(float(scores["test_f1_macro"].mean()), 4)
    }

    print("Accuracy:", results[name]["accuracy"])
    print("Precision (Weighted):", results[name]["precision"])
    print("Recall (Weighted):", results[name]["recall"])
    print("F1 (Weighted):", results[name]["f1"])
    print("F1 (Macro):", results[name]["f1_macro"])


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================
print("\n7. Selecting best model...")

best_model_name = max(
    results,
    key=lambda name: (
        results[name]["f1"],
        results[name]["accuracy"]
    )
)

print("Selected model:", best_model_name)


# ============================================================
# 12. TRAIN FINAL MODEL
# ============================================================
print("\n8. Training final model...")

final_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", models[best_model_name])
])

final_model.fit(X_train, y_train)
print("Final model trained")


# ============================================================
# 13. FINAL TEST
# ============================================================
print("\n9. Testing final model...")

predictions = final_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
confusion = confusion_matrix(y_test, predictions)

print("\nFinal Test Results")
print("-" * 40)
print("Accuracy:", round(accuracy, 4))
print("Precision (Weighted):", round(precision, 4))
print("Recall (Weighted):", round(recall, 4))
print("F1 Score (Weighted):", round(f1, 4))

print("\nConfusion Matrix:")
print(confusion)


# ============================================================
# 14. SAVE MODEL & METADATA
# ============================================================
print("\n10. Saving model...")
joblib.dump(final_model, MODEL_PATH)

metadata = {
    "best_model": best_model_name,
    "features": FEATURES,
    "target": TARGET,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "class_mapping": {"0": "Low", "1": "Medium", "2": "High"},
    "cross_validation_results": results,
    "test_metrics": {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion.tolist()
    },
    "selection_metric": "f1_weighted",
    "dataset": DATA_PATH.name
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print("\nModel saved:", MODEL_PATH)
print("Metadata saved:", METADATA_PATH)
print("\nTraining complete!")