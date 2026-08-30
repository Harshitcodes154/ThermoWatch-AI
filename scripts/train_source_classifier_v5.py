# ================================================================
# THERMO WATCH - AI SOURCE CLASSIFICATION MODEL V5
# ================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

warnings.filterwarnings("ignore")

# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "training_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "source_classifier_v5.joblib"
)

RESULT_PATH = os.path.join(
    OUTPUT_DIR,
    "model_v5_results.txt"
)

# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("THERMO WATCH - AI SOURCE CLASSIFICATION MODEL V5")
print("=" * 75)

# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading training dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows   : {len(df)}")
print(f"Columns: {len(df.columns)}")

# ================================================================
# REMOVE DUPLICATES
# ================================================================

if "event_id" in df.columns:

    before = len(df)

    df = df.drop_duplicates(
        subset=["event_id"]
    ).reset_index(drop=True)

    removed = before - len(df)

    print(f"Duplicate events removed: {removed}")

# ================================================================
# TARGET
# ================================================================

TARGET = "ai_label"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found!"
    )

print("\nTarget distribution:")
print(df[TARGET].value_counts())

# ================================================================
# ADVANCED FEATURE ENGINEERING
# ================================================================

print("\nEngineering V5 features...")

# ------------------------------
# Safe numeric conversions
# ------------------------------

numeric_base = [
    "mean_frp",
    "max_frp",
    "mean_brightness",
    "max_brightness",
    "observation_count",
    "persistence_days",
    "observations_per_day",
    "activity_density",
    "satellite_count",
    "distance_to_facility_km"
]

for col in numeric_base:

    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ================================================================
# INTENSITY FEATURES
# ================================================================

if {"mean_frp", "max_frp"}.issubset(df.columns):

    df["frp_ratio_v5"] = (
        df["max_frp"] /
        (df["mean_frp"] + 0.01)
    )

    df["frp_excess_v5"] = (
        df["max_frp"] -
        df["mean_frp"]
    )

    df["frp_log_v5"] = np.log1p(
        df["mean_frp"].clip(lower=0)
    )

    df["max_frp_log_v5"] = np.log1p(
        df["max_frp"].clip(lower=0)
    )

# ================================================================
# BRIGHTNESS FEATURES
# ================================================================

if {
    "mean_brightness",
    "max_brightness"
}.issubset(df.columns):

    df["brightness_range_v5"] = (
        df["max_brightness"] -
        df["mean_brightness"]
    )

    df["brightness_ratio_v5"] = (
        df["max_brightness"] /
        (df["mean_brightness"] + 0.01)
    )

    df["brightness_excess_v5"] = (
        df["max_brightness"] - 300
    )

# ================================================================
# PERSISTENCE FEATURES
# ================================================================

if "persistence_days" in df.columns:

    df["persistence_log_v5"] = np.log1p(
        df["persistence_days"].clip(lower=0)
    )

    df["persistence_months_v5"] = (
        df["persistence_days"] / 30
    )

    df["persistent_30d_v5"] = (
        df["persistence_days"] >= 30
    ).astype(int)

    df["persistent_90d_v5"] = (
        df["persistence_days"] >= 90
    ).astype(int)

    df["persistent_180d_v5"] = (
        df["persistence_days"] >= 180
    ).astype(int)

    df["persistent_270d_v5"] = (
        df["persistence_days"] >= 270
    ).astype(int)

# ================================================================
# OBSERVATION FEATURES
# ================================================================

if "observation_count" in df.columns:

    df["observation_log_v5"] = np.log1p(
        df["observation_count"].clip(lower=0)
    )

if {
    "observation_count",
    "persistence_days"
}.issubset(df.columns):

    df["obs_per_persistence_v5"] = (
        df["observation_count"] /
        (df["persistence_days"] + 1)
    )

# ================================================================
# ACTIVITY FEATURES
# ================================================================

if {
    "observations_per_day",
    "persistence_days"
}.issubset(df.columns):

    df["activity_persistence_v5"] = (
        df["observations_per_day"] *
        np.log1p(
            df["persistence_days"].clip(lower=0)
        )
    )

# ================================================================
# DISTANCE FEATURES
# ================================================================

if "distance_to_facility_km" in df.columns:

    distance = df["distance_to_facility_km"].clip(
        lower=0
    )

    df["distance_log_v5"] = np.log1p(distance)

    df["very_close_v5"] = (
        distance <= 1
    ).astype(int)

    df["within_2km_v5"] = (
        distance <= 2
    ).astype(int)

    df["within_5km_v5"] = (
        distance <= 5
    ).astype(int)

    df["within_10km_v5"] = (
        distance <= 10
    ).astype(int)

# ================================================================
# COMBINED SIGNAL FEATURES
# ================================================================

if {
    "mean_frp",
    "persistence_days"
}.issubset(df.columns):

    df["frp_persistence_v5"] = (
        df["mean_frp"] *
        np.log1p(
            df["persistence_days"].clip(lower=0)
        )
    )

if {
    "mean_frp",
    "distance_to_facility_km"
}.issubset(df.columns):

    df["frp_distance_signal_v5"] = (
        df["mean_frp"] /
        (df["distance_to_facility_km"] + 1)
    )

if {
    "observation_count",
    "distance_to_facility_km"
}.issubset(df.columns):

    df["activity_distance_signal_v5"] = (
        np.log1p(
            df["observation_count"].clip(lower=0)
        )
        /
        (df["distance_to_facility_km"] + 1)
    )

# ================================================================
# REMOVE IDENTIFIER / LEAKAGE FEATURES
# ================================================================

print("\nRemoving identifier / leakage columns...")

DROP_COLUMNS = [
    TARGET,
    "event_id",
    "osm_id",
    "facility_osm_id",

    # Existing labels / scores
    "source_type",
    "source_confidence",
    "source_confidence_score",
    "risk_category",
    "risk_score",

    # Facility identifiers
    "facility_name",
    "facility_operator",

    # Coordinates of the facility
    "facility_latitude",
    "facility_longitude",

    # Original coordinate identifiers
    "latitude",
    "longitude",

    # Dates
    "first_seen",
    "last_seen"
]

DROP_COLUMNS = [
    c for c in DROP_COLUMNS
    if c in df.columns
]

X = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)

y = df[TARGET]

# ================================================================
# REMOVE COMPLETELY EMPTY COLUMNS
# ================================================================

empty_columns = [
    col for col in X.columns
    if X[col].isna().all()
]

if empty_columns:

    print(
        "\nRemoving completely empty columns:"
    )

    for col in empty_columns:
        print(" -", col)

    X = X.drop(
        columns=empty_columns
    )

# ================================================================
# FEATURE TYPES
# ================================================================

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print(
    f"\nFinal feature count: {len(X.columns)}"
)

print(
    f"Numeric features   : {len(numeric_features)}"
)

print(
    f"Categorical features: {len(categorical_features)}"
)

print("\nFeatures:")

for feature in X.columns:
    print(" -", feature)

# ================================================================
# PREPROCESSING
# ================================================================

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
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)

# ================================================================
# TRAIN / TEST SPLIT
# ================================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training: {len(X_train)}"
)

print(
    f"Testing : {len(X_test)}"
)

# ================================================================
# MODELS
# ================================================================

models = {

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees_Deep": ExtraTreesClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# ================================================================
# TRAINING
# ================================================================

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

results = []

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
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

    balanced = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print(
        f"{name} Accuracy       : "
        f"{accuracy:.4f}"
    )

    print(
        f"{name} Balanced Acc.  : "
        f"{balanced:.4f}"
    )

    print(
        f"{name} Macro F1       : "
        f"{macro_f1:.4f}"
    )

    results.append(
        {
            "model": name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "macro_f1": macro_f1
        }
    )

    trained_models[name] = pipeline

# ================================================================
# SELECT BEST MODEL
# ================================================================

results_df = pd.DataFrame(
    results
)

# Primary metric = balanced accuracy
# Secondary metric = macro F1
# Third = accuracy

results_df = results_df.sort_values(
    by=[
        "balanced_accuracy",
        "macro_f1",
        "accuracy"
    ],
    ascending=False
).reset_index(drop=True)

best_name = results_df.iloc[0]["model"]

best_model = trained_models[
    best_name
]

best_predictions = best_model.predict(
    X_test
)

best_accuracy = accuracy_score(
    y_test,
    best_predictions
)

best_balanced = balanced_accuracy_score(
    y_test,
    best_predictions
)

best_macro_f1 = f1_score(
    y_test,
    best_predictions,
    average="macro"
)

# ================================================================
# BEST MODEL REPORT
# ================================================================

print("\n" + "=" * 75)
print("BEST MODEL")
print("=" * 75)

print(
    f"Model            : {best_name}"
)

print(
    f"Accuracy         : {best_accuracy:.4f}"
)

print(
    f"Balanced Accuracy: {best_balanced:.4f}"
)

print(
    f"Macro F1         : {best_macro_f1:.4f}"
)

print("\nClassification Report:")

report = classification_report(
    y_test,
    best_predictions
)

print(report)

# ================================================================
# CONFUSION MATRIX
# ================================================================

print("Confusion Matrix:")

labels = sorted(
    y.unique()
)

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

# ================================================================
# 5-FOLD CROSS VALIDATION
# ================================================================

print("\n" + "=" * 75)
print("5-FOLD CROSS VALIDATION")
print("=" * 75)

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
    scoring="balanced_accuracy",
    n_jobs=1
)

print("\nBalanced accuracy folds:")

for i, score in enumerate(
    cv_scores,
    start=1
):

    print(
        f"Fold {i}: {score:.4f}"
    )

print(
    f"\nCV Mean Balanced Accuracy: "
    f"{cv_scores.mean():.4f}"
)

print(
    f"CV Std                   : "
    f"{cv_scores.std():.4f}"
)

# ================================================================
# SAVE MODEL
# ================================================================

joblib.dump(
    best_model,
    MODEL_PATH
)

# ================================================================
# SAVE RESULTS
# ================================================================

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "THERMO WATCH - MODEL V5 RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Rows: {len(df)}\n"
    )

    f.write(
        f"Features: {len(X.columns)}\n\n"
    )

    f.write(
        "MODEL COMPARISON\n"
    )

    f.write(
        results_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        f"Best Model: {best_name}\n"
    )

    f.write(
        f"Accuracy: {best_accuracy:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy: "
        f"{best_balanced:.4f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{best_macro_f1:.4f}\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        report
    )

    f.write(
        "\nConfusion Matrix\n"
    )

    f.write(
        cm_df.to_string()
    )

    f.write(
        "\n\n5-Fold CV Balanced Accuracy\n"
    )

    for i, score in enumerate(
        cv_scores,
        start=1
    ):

        f.write(
            f"Fold {i}: {score:.4f}\n"
        )

    f.write(
        f"\nCV Mean: "
        f"{cv_scores.mean():.4f}\n"
    )

    f.write(
        f"CV Std: "
        f"{cv_scores.std():.4f}\n"
    )

# ================================================================
# SAMPLE PREDICTIONS
# ================================================================

print("\n" + "=" * 75)
print("SAMPLE PREDICTIONS")
print("=" * 75)

sample_count = min(
    10,
    len(X_test)
)

sample_X = X_test.iloc[
    :sample_count
]

sample_predictions = best_model.predict(
    sample_X
)

sample_probabilities = best_model.predict_proba(
    sample_X
)

classes = best_model.named_steps[
    "model"
].classes_

for i in range(sample_count):

    prediction = sample_predictions[i]

    confidence = (
        np.max(
            sample_probabilities[i]
        ) * 100
    )

    print(
        f"{i + 1:02d}. "
        f"{prediction:<20} "
        f"Confidence: "
        f"{confidence:.2f}%"
    )

# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 75)
print("MODEL V5 TRAINING COMPLETE")
print("=" * 75)

print(
    f"\nBest Model           : {best_name}"
)

print(
    f"Test Accuracy        : "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy    : "
    f"{best_balanced * 100:.2f}%"
)

print(
    f"Macro F1             : "
    f"{best_macro_f1 * 100:.2f}%"
)

print(
    f"CV Balanced Accuracy : "
    f"{cv_scores.mean() * 100:.2f}%"
)

print(
    f"\nSaved model:"
)

print(
    MODEL_PATH
)

print(
    "\nSaved results:"
)

print(
    RESULT_PATH
)

print(
    "\nNext step:"
)

print(
    "Use V5 model for LIVE hotspot source prediction."
)

print(
    "=" * 75
)