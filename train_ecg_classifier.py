"""Train the ECG escalation-flag classifier (PTBDB: normal vs abnormal).

Standalone companion to Preprocess_MLIoT_Dataset.ipynb's ECG pipeline (not an
edit of it — the notebook is a teammate's shared artifact). Trains on PTBDB
only, framed as binary normal/abnormal: this model's only job is to gate a
rule-based escalation flag (abnormal -> bump the IoMT triage tier by one),
not arrhythmia subtyping, which is why MIT-BIH's 5-class data isn't used here.
"""

from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import pyplot as plt

DATA_DIR = Path(__file__).parent / "data" / "heartbeat"
OUTPUT_DIR = Path(__file__).parent / "output"

WAVEFORM_LENGTH = 187
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2
RANDOM_STATE = 42
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 3


def load_ptbdb_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    normal = pd.read_csv(data_dir / "ptbdb_normal.csv", header=None)
    abnormal = pd.read_csv(data_dir / "ptbdb_abnormal.csv", header=None)
    combined = pd.concat([normal, abnormal], ignore_index=True)

    X = combined.iloc[:, :-1].to_numpy()
    y = combined.iloc[:, -1].to_numpy().astype(int)
    return X, y


def split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_FRACTION, stratify=y_train, random_state=RANDOM_STATE
    )

    return (
        X_train.reshape(-1, WAVEFORM_LENGTH, 1),
        X_val.reshape(-1, WAVEFORM_LENGTH, 1),
        X_test.reshape(-1, WAVEFORM_LENGTH, 1),
        y_train,
        y_val,
        y_test,
    )


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_model(input_length: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_length, 1)),
            layers.Conv1D(32, kernel_size=5, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, kernel_size=5, activation="relu"),
            layers.MaxPooling1D(2),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
) -> keras.callbacks.History:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, mode="max"
    )
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=2,
    )


def evaluate_model(
    model: keras.Model, X_test: np.ndarray, y_test: np.ndarray, output_dir: Path
) -> None:
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print(f"Test AUC: {auc:.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Normal", "Abnormal"], cmap="Blues", ax=ax
    )
    ax.set_title("ECG classifier (PTBDB) — confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "ecg_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"ECG classifier (PTBDB) — ROC curve (AUC = {auc:.3f})")
    fig.tight_layout()
    fig.savefig(output_dir / "ecg_roc_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    keras.utils.set_random_seed(RANDOM_STATE)

    X, y = load_ptbdb_data(DATA_DIR)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    class_weights = compute_class_weights(y_train)
    print("Class weights:", class_weights)

    model = build_model(WAVEFORM_LENGTH)
    model.summary()
    train_model(model, X_train, y_train, X_val, y_val, class_weights)

    evaluate_model(model, X_test, y_test, OUTPUT_DIR)

    model.save(OUTPUT_DIR / "ecg_cnn_model.keras")
    print(f"Saved model to {OUTPUT_DIR}/ecg_cnn_model.keras")


if __name__ == "__main__":
    main()
