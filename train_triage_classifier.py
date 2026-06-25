from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "iomt-alert" / "patients_data_with_alerts.xlsx"
OUTPUT_DIR = Path(__file__).parent / "output"

HR_COL = "Heart Rate (bpm)"
SPO2_COL = "SpO2 Level (%)"
SYSTOLIC_COL = "Systolic Blood Pressure (mmHg)"
DIASTOLIC_COL = "Diastolic Blood Pressure (mmHg)"
TEMP_COL = "Body Temperature (°C)"
FALL_COL = "Fall Detection"
TARGET_COL = "Triage"

# Features available on the actual ESP32 build (MAX30102 + DS18B20).
# Systolic/diastolic BP and Fall Detection are excluded. No BP cuff, no IMU.
FEATURE_COLUMNS = [HR_COL, SPO2_COL, TEMP_COL]

SMOTE_IMBALANCE_THRESHOLD = 1.5


def load_iomt_data(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def assign_triage(row: pd.Series) -> str:
    """Mirrors Preprocess_MLIoT_Dataset.ipynb's assign_triage() — kept in sync manually."""
    heart_rate = row[HR_COL]
    spo2 = row[SPO2_COL]
    systolic = row[SYSTOLIC_COL]
    diastolic = row[DIASTOLIC_COL]
    temperature = row[TEMP_COL]
    fall = str(row[FALL_COL]).strip().lower() == "yes"

    if (
        spo2 < 90
        or heart_rate < 40
        or heart_rate >= 130
        or systolic < 90
        or systolic >= 180
        or diastolic >= 120
        or temperature < 35
        or temperature >= 39.5
    ):
        return "Emergency"
    if (
        spo2 < 94
        or heart_rate < 50
        or heart_rate >= 110
        or systolic < 100
        or systolic >= 140
        or diastolic >= 90
        or temperature < 36
        or temperature >= 38
        or fall
    ):
        return "Urgent"
    return "Non-Urgent"


def build_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df[TARGET_COL] = df.apply(assign_triage, axis=1)
    df = df.drop_duplicates()

    X = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())
    y = df[TARGET_COL]
    return X, y


def encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder, dict[str, int]]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    label_mapping = {
        str(label): int(code) for label, code in zip(encoder.classes_, encoder.transform(encoder.classes_))
    }
    return y_encoded, encoder, label_mapping


def split_scale_balance(
    X: pd.DataFrame, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    class_counts = pd.Series(y_tr).value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > SMOTE_IMBALANCE_THRESHOLD:
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    return X_tr, X_te, y_tr, y_te, scaler


def train_model(X_tr: np.ndarray, y_tr: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_tr, y_tr)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_te: np.ndarray,
    y_te: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> None:
    y_pred = model.predict(X_te)
    class_names = label_encoder.classes_

    baseline_accuracy = pd.Series(y_te).value_counts(normalize=True).max()
    model_accuracy = (y_pred == y_te).mean()
    print(f"Majority-class baseline accuracy: {baseline_accuracy:.3f}")
    print(f"Model accuracy: {model_accuracy:.3f}")
    print()
    print(classification_report(y_te, y_pred, target_names=class_names))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred, display_labels=class_names, cmap="Blues", ax=ax
    )
    ax.set_title("Triage classifier — confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "triage_confusion_matrix.png", dpi=150)
    plt.close(fig)

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    print("Feature importances:")
    print(importances)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importances.index, importances.values, color="steelblue")
    ax.set_title("Triage classifier — feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "triage_feature_importance.png", dpi=150)
    plt.close(fig)


def save_artifacts(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    label_mapping: dict[str, int],
    output_dir: Path,
) -> None:
    joblib.dump(model, output_dir / "triage_rf_model.joblib")
    joblib.dump(scaler, output_dir / "triage_scaler.joblib")
    joblib.dump(label_mapping, output_dir / "triage_label_mapping.joblib")
    print(f"Saved model, scaler, and label mapping to {output_dir}/")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_iomt_data(DATA_PATH)
    X, y = build_features_and_target(df)
    y_encoded, label_encoder, label_mapping = encode_target(y)
    print("Label mapping:", label_mapping)

    X_tr, X_te, y_tr, y_te, scaler = split_scale_balance(X, y_encoded)
    model = train_model(X_tr, y_tr)
    evaluate_model(model, X_te, y_te, label_encoder, OUTPUT_DIR)
    save_artifacts(model, scaler, label_mapping, OUTPUT_DIR)


if __name__ == "__main__":
    main()
