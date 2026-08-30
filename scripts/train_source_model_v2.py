import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# THERMO WATCH - SOURCE CLASSIFICATION MODEL V2
# ============================================================

print("=" * 65)
print("THERMO WATCH - AI SOURCE CLASSIFICATION MODEL V2")
print("=" * 65)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "training_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading training dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

TARGET = "ai_label"

if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

df = df.dropna(subset=[TARGET]).copy()

print("\nTarget distribution:")
print(df[TARGET].value_counts())


# ============================================================
# IMPORTANT:
# Remove features that directly encode the old attribution
# decision. Otherwise the model can simply learn the old rule.
# ============================================================

NUMERIC_FEATURES = [
    "mean_frp",
    "max_frp",
    "frp_ratio",
    "mean_brightness",
    "max_brightness",
    "brightness_range",
    "observation_count",
    "persistence_days",
    "observations_per_day",
    "activity_density",
    "satellite_count",
    "distance_to_facility_km",
    "facility_within_1km",
    "facility_within_5km",
    "persistent_long_term",
    "persistent_180_days",
]


CATEGORICAL_FEATURES = [
    "facility_type",
    "facility_power",
    "facility_industrial",
    "facility_landuse",
]


NUMERIC_FEATURES = [
    c for c in NUMERIC_FEATURES
    if c in df.columns
]

CATEGORICAL_FEATURES = [
    c for c in CATEGORICAL_FEATURES
    if c in df.columns
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print("\nFeatures used:")
for feature in FEATURES:
    print(" -", feature)


X = df[FEATURES]
y = df[TARGET]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training: {len(X_train)}")
print(f"Testing : {len(X_test)}")


# ============================================================
# PREPROCESSOR
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN + COMPARE
# ============================================================

results = {}

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)


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

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = {
        "accuracy": accuracy,
        "pipeline": pipeline
    }

    print(
        f"{name} Accuracy: "
        f"{accuracy:.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

best_name = max(
    results,
    key=lambda x: results[x]["accuracy"]
)

best_model = results[best_name]["pipeline"]

best_accuracy = results[best_name]["accuracy"]


print("\n" + "=" * 65)
print("BEST MODEL")
print("=" * 65)

print(f"Model    : {best_name}")
print(f"Accuracy : {best_accuracy:.4f}")


# ============================================================
# DETAILED EVALUATION
# ============================================================

y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("5-FOLD CROSS VALIDATION")
print("=" * 65)

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\nFold accuracies:")

for i, score in enumerate(cv_scores, 1):
    print(
        f"Fold {i}: "
        f"{score:.4f}"
    )

print(
    f"\nCV Mean Accuracy: "
    f"{cv_scores.mean():.4f}"
)

print(
    f"CV Std: "
    f"{cv_scores.std():.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "source_classifier_v2.joblib"
)

joblib.dump(
    best_model,
    MODEL_FILE
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    BASE_DIR,
    "processed",
    "model_v2_results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "THERMO WATCH - MODEL V2 RESULTS\n"
    )

    f.write("=" * 50 + "\n")

    f.write(
        f"Best Model: {best_name}\n"
    )

    f.write(
        f"Test Accuracy: {best_accuracy:.4f}\n"
    )

    f.write(
        f"CV Mean Accuracy: "
        f"{cv_scores.mean():.4f}\n"
    )

    f.write(
        f"CV Std: "
        f"{cv_scores.std():.4f}\n"
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 65)
print("SAMPLE PREDICTIONS")
print("=" * 65)

sample = X_test.head(10)

sample_predictions = best_model.predict(sample)
sample_probabilities = best_model.predict_proba(sample)

classes = best_model.classes_

for i in range(len(sample)):

    prediction = sample_predictions[i]

    confidence = (
        sample_probabilities[i].max()
        * 100
    )

    print(
        f"{i + 1:02d}. "
        f"{prediction:<20} "
        f"Confidence: {confidence:.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("MODEL V2 TRAINING COMPLETE")
print("=" * 65)

print(
    f"\nSaved model:\n{MODEL_FILE}"
)

print(
    f"\nSaved results:\n{results_file}"
)

print("\nNext step:")
print("Use the V2 model for new hotspot predictions.")