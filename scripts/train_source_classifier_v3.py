import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

print("=" * 70)
print("THERMO WATCH - AI SOURCE CLASSIFICATION MODEL V3")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "training_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading training dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# ------------------------------------------------------------
# BASIC CLEANING
# ------------------------------------------------------------

df = df.copy()

# Remove duplicate event IDs if present
if "event_id" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["event_id"])
    print(f"Duplicate events removed: {before - len(df)}")

# Target
TARGET = "ai_label"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

df = df.dropna(subset=[TARGET])

print("\nTarget distribution:")
print(df[TARGET].value_counts())

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

print("\nEngineering additional features...")

# FRP-related features
if "mean_frp" in df.columns and "max_frp" in df.columns:

    df["frp_excess"] = (
        df["max_frp"] - df["mean_frp"]
    ).clip(lower=0)

    df["frp_log"] = np.log1p(
        df["mean_frp"].clip(lower=0)
    )

# Persistence features
if "persistence_days" in df.columns:

    df["persistence_log"] = np.log1p(
        df["persistence_days"].clip(lower=0)
    )

    df["long_persistence"] = (
        df["persistence_days"] >= 180
    ).astype(int)

# Observation density
if "observation_count" in df.columns:

    df["observation_log"] = np.log1p(
        df["observation_count"].clip(lower=0)
    )

# Distance transformations
if "distance_to_facility_km" in df.columns:

    df["distance_log"] = np.log1p(
        df["distance_to_facility_km"].clip(lower=0)
    )

    df["very_close_facility"] = (
        df["distance_to_facility_km"] <= 1
    ).astype(int)

    df["close_facility"] = (
        df["distance_to_facility_km"] <= 5
    ).astype(int)

# Brightness
if "mean_brightness" in df.columns and "max_brightness" in df.columns:

    df["brightness_excess"] = (
        df["max_brightness"] -
        df["mean_brightness"]
    ).clip(lower=0)

# Observation / persistence relationship
if (
    "observation_count" in df.columns
    and "persistence_days" in df.columns
):

    df["observations_per_persistence"] = (
        df["observation_count"] /
        df["persistence_days"].replace(0, np.nan)
    )

    df["observations_per_persistence"] = (
        df["observations_per_persistence"]
        .replace([np.inf, -np.inf], np.nan)
    )

# ------------------------------------------------------------
# REMOVE POTENTIAL LEAKAGE FEATURES
# ------------------------------------------------------------

print("\nRemoving target-leakage features...")

# These are either derived from the attribution itself
# or directly encode the label we are trying to predict.

DROP_COLUMNS = [
    TARGET,

    # Existing source attribution
    "source_type",
    "source_confidence",
    "source_confidence_score",

    # Risk score/category are downstream engineered outputs
    "risk_score",
    "risk_category",

    # Facility operator/name can act like a memorized label
    # rather than generalizable evidence.
    "facility_operator",
    "facility_name",

    # Event ID is an identifier, not a predictive feature
    "event_id",

    # Coordinates can cause geographic memorization
    # and artificially optimistic random-split accuracy.
    "latitude",
    "longitude",
]

DROP_COLUMNS = [
    c for c in DROP_COLUMNS
    if c in df.columns
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]

print(f"Final feature count: {len(X.columns)}")

print("\nFeatures:")
for col in X.columns:
    print(" -", col)

# ------------------------------------------------------------
# IDENTIFY NUMERIC / CATEGORICAL
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training:", len(X_train))
print("Testing :", len(X_test))

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
}

# ------------------------------------------------------------
# TRAIN + EVALUATE
# ------------------------------------------------------------

results = []

trained_models = {}

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        predictions
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc
    })

    trained_models[name] = pipeline

    print(f"{name} Accuracy       : {accuracy:.4f}")
    print(f"{name} Balanced Acc.  : {balanced_acc:.4f}")

# ------------------------------------------------------------
# BEST MODEL
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

best_name = results_df.sort_values(
    by="balanced_accuracy",
    ascending=False
).iloc[0]["model"]

best_model = trained_models[best_name]

best_predictions = best_model.predict(X_test)

best_accuracy = accuracy_score(
    y_test,
    best_predictions
)

best_balanced = balanced_accuracy_score(
    y_test,
    best_predictions
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model            :", best_name)
print(f"Accuracy         : {best_accuracy:.4f}")
print(f"Balanced Accuracy: {best_balanced:.4f}")

# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        best_predictions,
        digits=4
    )
)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

print("\nConfusion Matrix:")

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)

# ------------------------------------------------------------
# 5-FOLD CROSS VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\nFold accuracies:")

for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")

print(
    f"\nCV Mean Accuracy: {cv_scores.mean():.4f}"
)

print(
    f"CV Std          : {cv_scores.std():.4f}"
)

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "source_classifier_v3.joblib"
)

joblib.dump(
    best_model,
    model_path
)

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_path = os.path.join(
    PROCESSED_DIR,
    "model_v3_results.txt"
)

with open(results_path, "w", encoding="utf-8") as f:

    f.write(
        "THERMO WATCH - SOURCE CLASSIFICATION V3\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Best Model: {best_name}\n"
    )

    f.write(
        f"Test Accuracy: {best_accuracy:.6f}\n"
    )

    f.write(
        f"Balanced Accuracy: {best_balanced:.6f}\n"
    )

    f.write(
        f"CV Mean Accuracy: {cv_scores.mean():.6f}\n"
    )

    f.write(
        f"CV Std: {cv_scores.std():.6f}\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(
        classification_report(
            y_test,
            best_predictions,
            digits=4
        )
    )

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm_df))

print("\n" + "=" * 70)
print("MODEL V3 TRAINING COMPLETE")
print("=" * 70)

print("\nSaved model:")
print(model_path)

print("\nSaved results:")
print(results_path)

print("\nNext step:")
print("Evaluate V3 against V2 and then build live hotspot inference.")