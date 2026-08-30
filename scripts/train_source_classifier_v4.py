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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 72)
print("THERMO WATCH - AI SOURCE CLASSIFICATION MODEL V4")
print("=" * 72)

# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "training_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# LOAD DATA
# ==============================================================

print("\nLoading training dataset...")

df = pd.read_csv(DATA_PATH)

print("Rows   :", len(df))
print("Columns:", len(df.columns))

TARGET = "ai_label"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

# Remove duplicate events
before = len(df)

if "event_id" in df.columns:
    df = df.drop_duplicates(
        subset=["event_id"]
    )

print(
    "Duplicate events removed:",
    before - len(df)
)

df = df.dropna(subset=[TARGET])

print("\nTarget distribution:")
print(df[TARGET].value_counts())

# ==============================================================
# FEATURE ENGINEERING
# ==============================================================

print("\nCreating advanced attribution features...")

# --------------------------------------------------------------
# Fire intensity
# --------------------------------------------------------------

if {"mean_frp", "max_frp"}.issubset(df.columns):

    df["frp_excess"] = (
        df["max_frp"] -
        df["mean_frp"]
    ).clip(lower=0)

    df["frp_ratio_safe"] = (
        df["max_frp"] /
        df["mean_frp"].replace(0, np.nan)
    )

    df["frp_log"] = np.log1p(
        df["mean_frp"].clip(lower=0)
    )

# --------------------------------------------------------------
# Brightness
# --------------------------------------------------------------

if {"mean_brightness", "max_brightness"}.issubset(df.columns):

    df["brightness_excess"] = (
        df["max_brightness"] -
        df["mean_brightness"]
    ).clip(lower=0)

    df["brightness_ratio"] = (
        df["max_brightness"] /
        df["mean_brightness"].replace(0, np.nan)
    )

# --------------------------------------------------------------
# Persistence
# --------------------------------------------------------------

if "persistence_days" in df.columns:

    df["persistence_log"] = np.log1p(
        df["persistence_days"].clip(lower=0)
    )

    df["persistence_months"] = (
        df["persistence_days"] / 30.0
    )

    df["persistent_30d"] = (
        df["persistence_days"] >= 30
    ).astype(int)

    df["persistent_90d"] = (
        df["persistence_days"] >= 90
    ).astype(int)

    df["persistent_180d"] = (
        df["persistence_days"] >= 180
    ).astype(int)

# --------------------------------------------------------------
# Observation behaviour
# --------------------------------------------------------------

if "observation_count" in df.columns:

    df["observation_log"] = np.log1p(
        df["observation_count"].clip(lower=0)
    )

if {
    "observation_count",
    "persistence_days"
}.issubset(df.columns):

    df["obs_per_day_calc"] = (
        df["observation_count"] /
        df["persistence_days"].replace(0, np.nan)
    )

# --------------------------------------------------------------
# Facility proximity
# --------------------------------------------------------------

if "distance_to_facility_km" in df.columns:

    df["distance_log"] = np.log1p(
        df["distance_to_facility_km"].clip(lower=0)
    )

    df["very_close"] = (
        df["distance_to_facility_km"] <= 1
    ).astype(int)

    df["within_2km"] = (
        df["distance_to_facility_km"] <= 2
    ).astype(int)

    df["within_5km"] = (
        df["distance_to_facility_km"] <= 5
    ).astype(int)

    df["within_10km"] = (
        df["distance_to_facility_km"] <= 10
    ).astype(int)

# --------------------------------------------------------------
# Interaction features
# --------------------------------------------------------------

if {
    "mean_frp",
    "persistence_days"
}.issubset(df.columns):

    df["frp_persistence"] = (
        df["mean_frp"] *
        np.log1p(
            df["persistence_days"].clip(lower=0)
        )
    )

if {
    "max_frp",
    "distance_to_facility_km"
}.issubset(df.columns):

    df["frp_distance_signal"] = (
        df["max_frp"] /
        (1.0 + df["distance_to_facility_km"])
    )

if {
    "mean_frp",
    "observation_count"
}.issubset(df.columns):

    df["activity_signal"] = (
        df["mean_frp"] *
        np.log1p(
            df["observation_count"].clip(lower=0)
        )
    )

# ==============================================================
# REMOVE LEAKAGE
# ==============================================================

print("\nRemoving leakage / identifier columns...")

DROP_COLUMNS = [
    TARGET,

    # Directly derived source information
    "source_type",
    "source_confidence",
    "source_confidence_score",

    # Downstream risk outputs
    "risk_score",
    "risk_category",

    # Identifier
    "event_id",

    # Geographic memorization
    "latitude",
    "longitude",

    # Avoid memorizing individual facilities
    "facility_name",
    "facility_operator",

    # Facility coordinates are redundant with distance
    "facility_latitude",
    "facility_longitude",
    "facility_osm_id",
]

DROP_COLUMNS = [
    c for c in DROP_COLUMNS
    if c in df.columns
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]

# ==============================================================
# CLEAN INF / NAN
# ==============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

print("\nFinal feature count:", len(X.columns))

print("\nFeatures:")

for feature in X.columns:
    print(" -", feature)

# ==============================================================
# DATA TYPES
# ==============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()

print(
    "\nNumeric features:",
    len(numeric_features)
)

print(
    "Categorical features:",
    len(categorical_features)
)

# ==============================================================
# PREPROCESSING
# ==============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
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
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)

# ==============================================================
# TRAIN / TEST
# ==============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training:", len(X_train))
print("Testing :", len(X_test))

# ==============================================================
# MODELS
# ==============================================================

models = {

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=800,
        max_features="sqrt",
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=800,
        max_features="sqrt",
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
}

# ==============================================================
# TRAINING
# ==============================================================

results = []
trained_models = {}

print("\n" + "=" * 72)
print("MODEL COMPARISON")
print("=" * 72)

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy
    })

    trained_models[name] = pipeline

    print(
        f"{name} Accuracy       : "
        f"{accuracy:.4f}"
    )

    print(
        f"{name} Balanced Acc.  : "
        f"{balanced_accuracy:.4f}"
    )

# ==============================================================
# SELECT BEST
# ==============================================================

results_df = pd.DataFrame(results)

best_row = results_df.sort_values(
    by=[
        "balanced_accuracy",
        "accuracy"
    ],
    ascending=False
).iloc[0]

best_name = best_row["model"]

best_model = trained_models[
    best_name
]

predictions = best_model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

# ==============================================================
# REPORT
# ==============================================================

print("\n" + "=" * 72)
print("BEST MODEL")
print("=" * 72)

print("Model            :", best_name)
print(f"Accuracy         : {accuracy:.4f}")
print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy:.4f}"
)

print("\nClassification Report:")

report = classification_report(
    y_test,
    predictions,
    digits=4
)

print(report)

# ==============================================================
# CONFUSION MATRIX
# ==============================================================

labels = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nConfusion Matrix:")
print(cm_df)

# ==============================================================
# CROSS VALIDATION
# ==============================================================

print("\n" + "=" * 72)
print("5-FOLD CROSS VALIDATION")
print("=" * 72)

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

for i, score in enumerate(
    cv_scores,
    start=1
):
    print(
        f"Fold {i}: {score:.4f}"
    )

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(
    f"\nCV Mean Accuracy: "
    f"{cv_mean:.4f}"
)

print(
    f"CV Std          : "
    f"{cv_std:.4f}"
)

# ==============================================================
# SAVE MODEL
# ==============================================================

model_path = os.path.join(
    MODEL_DIR,
    "source_classifier_v4.joblib"
)

joblib.dump(
    best_model,
    model_path
)

# ==============================================================
# SAVE RESULTS
# ==============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "model_v4_results.txt"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "THERMO WATCH - SOURCE CLASSIFICATION V4\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Best Model: {best_name}\n"
    )

    f.write(
        f"Test Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.6f}\n"
    )

    f.write(
        f"CV Mean Accuracy: "
        f"{cv_mean:.6f}\n"
    )

    f.write(
        f"CV Std: {cv_std:.6f}\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(report)

    f.write(
        "\n\nConfusion Matrix:\n"
    )

    f.write(
        cm_df.to_string()
    )

print("\n" + "=" * 72)
print("MODEL V4 TRAINING COMPLETE")
print("=" * 72)

print("\nSaved model:")
print(model_path)

print("\nSaved results:")
print(results_path)

print("\nNext:")
print(
    "Use V4 for source attribution "
    "and then build live hotspot inference."
)