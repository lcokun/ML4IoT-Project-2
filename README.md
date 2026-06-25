# IoMT Triage Severity Classification — ML Pipeline

SAIA3353 ML4IoT final project. This covers the Data Processing & Intelligence Engineer scope: preprocessing, triage labelling, model training, and the ECG escalation rule. Hardware (ESP32 + sensors) and the Node-RED/MQTT/Streamlit pipeline are teammates' scope — see `context.md` for the full system picture and design rationale.

## What this does

Takes patient vitals (heart rate, SpO2, body temperature) and outputs a 3-tier triage decision (Non-Urgent / Urgent / Emergency, i.e. MOH P3/P2/P1), escalated by one tier if a paired ECG reading is flagged abnormal.

## Setup

From the repo root (`ML4IoT-SAIA3353/`):

```bash
uv sync
```

## Files

```
final-project/
├── context.md                      # full design history, decisions, and rationale — read this for "why"
├── data/
│   ├── iomt-alert/                 # vitals dataset (patients_data_with_alerts.xlsx)
│   └── heartbeat/                  # ECG dataset (MIT-BIH + PTBDB)
├── visualize_iomt_thresholds.py    # diagnostic: which triage thresholds actually fire in the real data
├── train_triage_classifier.py      # trains the triage RF model (HR/SpO2/temp -> tier)
├── train_ecg_classifier.py         # trains the ECG CNN (PTBDB binary: normal/abnormal)
├── triage_fusion.py                # combines both models: ECG abnormal -> escalate tier by one
├── output/                         # trained models, plots, evaluation artifacts
├── Preprocess_MLIoT_Dataset.ipynb  # teammate's notebook — reference only, not edited
└── inspect_data.ipynb
```

## Running the pipeline

From `final-project/`:

```bash
uv run python3 train_triage_classifier.py    # trains + saves the triage RF model
uv run python3 train_ecg_classifier.py       # trains + saves the ECG CNN
uv run python3 triage_fusion.py              # demo: combines both models' outputs
uv run python3 visualize_iomt_thresholds.py  # diagnostic plots only, no training
```

Each script is standalone and writes its artifacts to `output/`.

## Status

- Triage classifier, ECG classifier, and fusion/escalation logic: done and verified (see `context.md` for metrics, an independent audit log, and known limitations).
- Inference packaging (taking live sensor input over MQTT and publishing a decision) — not yet built. This is the only remaining piece, and needs a topology decision from the hardware team first (where Node-RED/Mosquitto/Streamlit actually run).

## Known limitations (see `context.md` for detail)

- Triage labels are synthetic — manufactured from raw vitals via clinically-informed thresholds, not a real ground-truth column. HR/BP/fall cutoffs are checked against a medical student's input plus a standard BP staging chart; SpO2/temp cutoffs are not independently validated.
- Blood pressure and fall detection are used to *generate* the triage label but excluded from the *classifier's* features — no BP cuff or IMU on the real ESP32 build. This costs real accuracy on the Non-Urgent tier specifically (0.18 precision / 0.30 recall).
- The "obviously healthy" demo case in `triage_fusion.py` is a real dataset row, not hand-picked synthetic numbers — made-up "healthy" vitals don't reliably classify as Non-Urgent.
