"""Fusion/escalation logic: combine the triage classifier and ECG classifier.

Not a third trained model — a rule layered on top of the two independently
trained models in train_triage_classifier.py and train_ecg_classifier.py:
if the ECG classifier flags "abnormal", escalate the triage tier by one
level (Non-Urgent -> Urgent -> Emergency; Emergency is already the top tier
and stays there).
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent / "output"
ECG_DATA_DIR = Path(__file__).parent / "data" / "heartbeat"

TRIAGE_MODEL_PATH = OUTPUT_DIR / "triage_rf_model.joblib"
TRIAGE_SCALER_PATH = OUTPUT_DIR / "triage_scaler.joblib"
TRIAGE_LABEL_MAPPING_PATH = OUTPUT_DIR / "triage_label_mapping.joblib"
ECG_MODEL_PATH = OUTPUT_DIR / "ecg_cnn_model.keras"

WAVEFORM_LENGTH = 187
ECG_ABNORMAL_THRESHOLD = 0.5

# Must match train_triage_classifier.py's FEATURE_COLUMNS order — the scaler
# was fit on a DataFrame with these column names.
TRIAGE_FEATURE_COLUMNS = ["Heart Rate (bpm)", "SpO2 Level (%)", "Body Temperature (°C)"]

# Increasing severity. Mirrors context.md's P3/P2/P1 = Non-Urgent/Urgent/Emergency.
TIER_ORDER = ["Non-Urgent", "Urgent", "Emergency"]


@dataclass
class TriageDecision:
    base_tier: str
    ecg_abnormal: bool
    final_tier: str


def load_triage_artifacts() -> tuple[RandomForestClassifier, StandardScaler, dict[str, int]]:
    model = joblib.load(TRIAGE_MODEL_PATH)
    scaler = joblib.load(TRIAGE_SCALER_PATH)
    label_mapping = joblib.load(TRIAGE_LABEL_MAPPING_PATH)
    return model, scaler, label_mapping


def load_ecg_model() -> keras.Model:
    return keras.models.load_model(ECG_MODEL_PATH)


def predict_triage_tier(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    label_mapping: dict[str, int],
    hr: float,
    spo2: float,
    temp: float,
) -> str:
    code_to_label = {code: label for label, code in label_mapping.items()}
    features = pd.DataFrame([[hr, spo2, temp]], columns=TRIAGE_FEATURE_COLUMNS)
    predicted_code = model.predict(scaler.transform(features))[0]
    return code_to_label[predicted_code]


def predict_ecg_abnormal(model: keras.Model, waveform: np.ndarray) -> bool:
    reshaped = waveform.reshape(1, WAVEFORM_LENGTH, 1)
    probability = model.predict(reshaped, verbose=0)[0, 0]
    return probability >= ECG_ABNORMAL_THRESHOLD


def escalate_tier(tier: str, ecg_abnormal: bool) -> str:
    if not ecg_abnormal:
        return tier
    next_index = min(TIER_ORDER.index(tier) + 1, len(TIER_ORDER) - 1)
    return TIER_ORDER[next_index]


def fuse_triage_decision(
    triage_model: RandomForestClassifier,
    triage_scaler: StandardScaler,
    triage_label_mapping: dict[str, int],
    ecg_model: keras.Model,
    hr: float,
    spo2: float,
    temp: float,
    ecg_waveform: np.ndarray,
) -> TriageDecision:
    base_tier = predict_triage_tier(triage_model, triage_scaler, triage_label_mapping, hr, spo2, temp)
    ecg_abnormal = predict_ecg_abnormal(ecg_model, ecg_waveform)
    final_tier = escalate_tier(base_tier, ecg_abnormal)
    return TriageDecision(base_tier=base_tier, ecg_abnormal=bool(ecg_abnormal), final_tier=final_tier)


def load_example_waveform(filename: str) -> np.ndarray:
    row = pd.read_csv(ECG_DATA_DIR / filename, header=None, nrows=1)
    return row.iloc[0, :-1].to_numpy(dtype=float)


def main() -> None:
    triage_model, triage_scaler, triage_label_mapping = load_triage_artifacts()
    ecg_model = load_ecg_model()

    normal_waveform = load_example_waveform("ptbdb_normal.csv")
    abnormal_waveform = load_example_waveform("ptbdb_abnormal.csv")

    # Illustrative pairings only — the IoMT vitals and ECG waveforms come from
    # unrelated datasets with no shared patient ID (see context.md), so these
    # are not real paired patients, just representative readings per tier.
    demo_cases = [
        ("Non-Urgent vitals + normal ECG -> no escalation", 99, 97, 36.46, normal_waveform),
        ("Non-Urgent vitals + abnormal ECG -> escalates one tier", 99, 97, 36.46, abnormal_waveform),
        ("Urgent vitals + abnormal ECG -> escalates one tier", 105, 92, 37.2, abnormal_waveform),
        ("Emergency vitals + abnormal ECG -> capped, stays Emergency", 140, 87, 37.0, abnormal_waveform),
    ]

    for description, hr, spo2, temp, waveform in demo_cases:
        decision = fuse_triage_decision(
            triage_model, triage_scaler, triage_label_mapping, ecg_model, hr, spo2, temp, waveform
        )
        print(f"{description}")
        print(
            f"  base_tier={decision.base_tier}  ecg_abnormal={decision.ecg_abnormal}  "
            f"final_tier={decision.final_tier}"
        )


if __name__ == "__main__":
    main()
