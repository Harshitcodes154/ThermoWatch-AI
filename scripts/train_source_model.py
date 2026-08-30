import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ============================================================
# THERMO WATCH - AI SOURCE CLASSIFICATION MODEL
# ============================================================

print("=" * 60)
print("THERMO WATCH - AI SOURCE CLASSIFICATION")
print("=" * 60)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "training_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "source_classifier.joblib"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading training dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

TARGET = "ai_label"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found in dataset."
    )


print("\nTarget distribution:")
print(df[TARGET].value_counts())


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

# Numerical features describing thermal activity,
# persistence, proximity and risk.

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
    "source_confidence_score",
    "risk_score",
]


# Categorical/context features.

CATEGORICAL_FEATURES = [
    "facility_type",
    "facility_power",
    "facility_industrial",
    "facility_landuse",
    "source_confidence",
    "risk_category",
]


# Keep only columns that actually exist.
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
print(FEATURES)


# ------------------------------------------------------------
# CLEAN TARGET
# ------------------------------------------------------------

df = df.dropna(subset=[TARGET]).copy()

X = df[FEATURES]
y = df[TARGET]


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

print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
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
        ),
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
        ),
    ]
)


# ------------------------------------------------------------
# RANDOM FOREST MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        ),
    ]
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy: {accuracy:.4f}")

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


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

joblib.dump(
    pipeline,
    MODEL_FILE
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"\nModel:")
print(MODEL_FILE)


# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

print("\nSample predictions:")

sample = X_test.head(10)

predictions = pipeline.predict(sample)

probabilities = pipeline.predict_proba(sample)

classes = pipeline.classes_

for i, prediction in enumerate(predictions):

    confidence = probabilities[i].max() * 100

    print(
        f"{i + 1}. "
        f"Prediction: {prediction} "
        f"| Confidence: {confidence:.2f}%"
    )


print("\n" + "=" * 60)
print("AI SOURCE CLASSIFICATION COMPLETE")
print("=" * 60)