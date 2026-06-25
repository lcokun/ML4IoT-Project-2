"""Loads the trained triage RF + ECG CNN once and exposes a single
evaluate() entry point. This is a thin wrapper around the same logic in
triage_fusion.py (predict_triage_tier / predict_ecg_abnormal / escalate_tier)
-- it does not retrain or redefine the models, just reuses them so the
Streamlit app and the MQTT listener don't each load a 100+MB Keras model
from scratch on every rerun.

If the trained artifacts aren't present yet (output/ is empty because
train_triage_classifier.py / train_ecg_classifier.py haven't been run),
this degrades to a clearly-labeled "model not loaded" state rather than
crashing, so the UI is still inspectable before training finishes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# triage_fusion.py / train_triage_classifier.py live alongside this file
# when dropped into the same project root as the uploaded scripts.
sys.path.insert(0, str(Path(__file__).parent))

WAVEFORM_LENGTH = 187
TIER_ORDER = ["Non-Urgent", "Urgent", "Emergency"]

TIER_COLORS = {
    "Non-Urgent": "#2e7d32",   # green
    "Urgent": "#f9a825",       # amber
    "Emergency": "#c62828",    # red
}


@dataclass
class TriageResult:
    base_tier: str
    ecg_abnormal: bool
    ecg_probability: float
    final_tier: str
    escalated: bool


class TriageEngine:
    """Lazily loads model artifacts on first use, then caches them."""

    def __init__(self) -> None:
        self._triage_model = None
        self._triage_scaler = None
        self._triage_label_mapping = None
        self._ecg_model = None
        self.load_error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return (
            self._triage_model is not None
            and self._triage_scaler is not None
            and self._triage_label_mapping is not None
            and self._ecg_model is not None
        )

    def load(self) -> bool:
        """Attempt to load all artifacts. Returns True on success.

        Safe to call repeatedly (e.g. from a Streamlit rerun) -- it's a
        no-op once already loaded, and re-attempts if it previously failed
        (so you can train the models after starting the app and just hit
        retry instead of restarting the process).
        """
        if self.is_ready:
            return True

        try:
            from triage_fusion import (  # noqa: WPS433 (local import by design)
                load_triage_artifacts,
                load_ecg_model,
            )

            self._triage_model, self._triage_scaler, self._triage_label_mapping = (
                load_triage_artifacts()
            )
            self._ecg_model = load_ecg_model()
            self.load_error = None
            return True
        except FileNotFoundError as exc:
            self.load_error = (
                "Trained model artifacts not found. Run train_triage_classifier.py "
                f"and train_ecg_classifier.py first to produce output/. ({exc})"
            )
            return False
        except Exception as exc:  # noqa: BLE001 - surface any load issue to the UI
            self.load_error = f"Failed to load models: {exc}"
            return False

    def evaluate(self, hr: float, spo2: float, temp: float, ecg_waveform: np.ndarray) -> TriageResult:
        if not self.is_ready:
            raise RuntimeError("TriageEngine.load() must succeed before evaluate() is called.")

        from triage_fusion import (  # noqa: WPS433
            predict_triage_tier,
            predict_ecg_abnormal,
            escalate_tier,
        )

        base_tier = predict_triage_tier(
            self._triage_model, self._triage_scaler, self._triage_label_mapping, hr, spo2, temp
        )

        reshaped = np.asarray(ecg_waveform, dtype=float).reshape(1, WAVEFORM_LENGTH, 1)
        probability = float(self._ecg_model.predict(reshaped, verbose=0)[0, 0])
        ecg_abnormal = probability >= 0.5

        final_tier = escalate_tier(base_tier, ecg_abnormal)

        return TriageResult(
            base_tier=base_tier,
            ecg_abnormal=ecg_abnormal,
            ecg_probability=probability,
            final_tier=final_tier,
            escalated=final_tier != base_tier,
        )


# Module-level singleton so Streamlit's rerun-on-every-interaction model
# doesn't reload the Keras model from disk each time. Streamlit's own
# @st.cache_resource is applied around this in streamlit_app.py.
engine = TriageEngine()
