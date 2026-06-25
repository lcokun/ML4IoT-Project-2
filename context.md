# Project Context: IoMT Triage Severity Classification System

## Project Identity

University group project for SAIA3353 Machine Learning for IoT (ML4IoT), UTM.
Assignment: "Hardware-Based IoT System Implementation" — hardware demo + presentation, no written report.
Presentation/demo date: 26 June 2026.

This document covers **Student 4's scope**: Data Processing and Intelligence Engineer.
That role = preprocessing, ML model design, decision logic, evaluation. Not hardware wiring, not the Node-RED/Streamlit/MQTT plumbing (those are separate teammates' scope, mentioned here only for interface context).

## The Problem Statement

Build a patient triage severity classifier that takes vitals from physical sensors and outputs a severity tier on **Malaysia's 3-tier MOH triage scale**:

- **P1** — Emergent (life-threatening, immediate attention)
- **P2** — Urgent (needs prompt attention, not immediately life-threatening)
- **P3** — Non-urgent (stable, can wait)

This 3-tier scale is the target system. It's a simplified collapse of the 5-level RETTS (Rapid Emergency Triage and Treatment System) used as international reference: RETTS red+orange → P1, yellow → P2, green+blue → P3.

## Hardware (for context — built by other teammates)

- ESP32 (confirmed final choice — NOT Raspberry Pi, despite earlier planning discussion assuming Pi)
- MAX30102 — heart rate (bpm) + SpO2 (%), I2C
- DS18B20 — body temperature (°C), 1-Wire protocol, needs 4.7kΩ pull-up resistor
- AD8232 + ADC — ECG signal, analog. ESP32 has a **built-in ADC** (unlike Raspberry Pi), so no external MCP3008 module is needed — confirm pin choice avoids ESP32's known noisy/non-linear ADC channels (ADC2 conflicts with WiFi; prefer ADC1 pins) for cleaner ECG sampling.
- Respiration sensor: **dropped**, not in scope. Do not build for it.

Data pipeline (other teammates' scope, relevant for output interface):
Sensors → ESP32 → MQTT (local Mosquitto broker) → Node-RED (decision logic integration point) → MQTT → Streamlit (live dashboard).
No cloud dependency, no Favoriot (explicitly not using Favoriot despite it being mentioned in the assignment brief as a suggestion).

**Where Node-RED/Streamlit/Mosquitto actually run is now an open question that needs confirming** — earlier planning assumed everything ran locally on a Raspberry Pi, but with ESP32 as the sensor node, the broker/Node-RED/Streamlit stack likely needs to run on a separate machine (laptop, or a Pi used purely as a server with no sensors attached) since ESP32 itself can't host Mosquitto/Node-RED/Streamlit. Confirm this topology before assuming where ML inference is actually deployed.

**Implication for this scope:** ESP32 has far less compute/memory than a Pi (no OS, microcontroller-class). Running the trained ML model directly on the ESP32 is unrealistic for anything beyond a trivial model. Realistic split: ESP32 handles sensor reading + publishing raw values over MQTT only; the actual triage classifier and ECG model run as inference on whatever machine hosts Node-RED (laptop or server-mode Pi), not on the ESP32 itself. Confirm this with the firmware/hardware teammates before building inference packaging in step 6 below.

## Datasets In Use

### 1. IoMT Dataset for ML-Based Health Monitoring (Kaggle, prokashbarmancu/iomt-alert)
File: `patients_data_with_alerts.xlsx`, 50,000 rows, 13 columns.

Columns:
| Column | Type | Notes |
|---|---|---|
| Patient Number | int | identifier, drop |
| Heart Rate (bpm) | int | raw — **keep as feature** |
| SpO2 Level (%) | int | raw — **keep as feature** |
| Systolic Blood Pressure (mmHg) | int | raw — keep as feature (no matching sensor, but useful if available; handle missing gracefully at inference) |
| Diastolic Blood Pressure (mmHg) | int | raw — same as above |
| Body Temperature (°C) | float | raw — **keep as feature** |
| Fall Detection | str (Yes/No) | optional feature, no matching sensor currently, low priority |
| Predicted Disease | str | Diabetes Mellitus / Asthma / Normal / Hypertension / Arrhythmia / etc. **DO NOT USE AS TARGET.** This is a diagnosis label, unrelated to triage severity. Likely itself a derived/synthetic label from the dataset generator, not real clinical ground truth. Drop entirely or ignore. |
| Data Accuracy (%) | int | likely a synthetic-data-generation artifact, not a real sensor confidence score. Drop — do not feed into training, since real Pi sensors won't produce this field at inference. |
| Heart Rate Alert | str (Normal/High) | **Verified pure threshold rule**: High ⟺ HR > 100. Confirmed via groupby min/max — clean non-overlapping ranges. Do not use as ML target (trivially recoverable via threshold, teaches nothing). Useful only as a sanity-check baseline. |
| SpO2 Level Alert | str (Normal/Low) | **Verified pure threshold rule**: Low ⟺ SpO2 < 90. Same situation as above. |
| Blood Pressure Alert | str (Normal/High) | Not independently verified, likely also a threshold rule. Treat with same suspicion. |
| Temperature Alert | str (Normal/Abnormal) | **Verified BROKEN/UNRELIABLE.** Abnormal range overlaps almost entirely with Normal range (Abnormal: 36.0–38.0°C, Normal: 36.1–37.2°C). A temp of 36.5°C could be labeled either way depending on row. **Do not trust this column for anything.** Do not use it to derive ground truth, do not use it as a feature, do not validate against it. |

**Critical finding from prior analysis:** none of the four "Alert" columns are usable as a triage severity target. They're either trivial threshold restatements (HR, SpO2) or actively broken (Temperature). The dataset has **no real severity ground truth column**. This was cross-checked against a published Kaggle notebook on the same dataset (vaibhavsatish/iomt-data) which independently hit the same wall and explicitly added a "Phase 2: Clinically Informed Synthetic Label Generation" step — i.e., even other analysts of this exact dataset had to manufacture their own severity labels. This validates the approach below; it is not a workaround unique to this project.

### 2. ECG Heartbeat Categorization Dataset (Kaggle, shayanfazeli/heartbeat)
Files: `mitbih_train.csv`, `mitbih_test.csv`, `ptbdb_normal.csv`, `ptbdb_abnormal.csv`.

- 187 normalized waveform sample columns (no header in raw file — **must load with `header=None`**, otherwise the first real data row gets silently consumed as a header) + 1 label column.
- MIT-BIH: multi-class arrhythmia beat classification (classes are integer-encoded beat types — normal, supraventricular, ventricular, fusion, unclassifiable).
- PTBDB: binary normal (0.0) vs abnormal/MI (1.0) classification.
- This is a **separate modeling problem from the IoMT vitals pipeline.** It classifies ECG waveform morphology, not overall triage severity, and has no shared patient ID or label scheme with the IoMT dataset.
- **Decision made:** this will NOT be fused into a single multi-input model with the IoMT vitals data. Instead, the ECG model's output (normal/abnormal classification) will act as a **severity escalation flag** — e.g., if ECG flags "abnormal," bump the IoMT-vitals-derived triage tier up by one level (P3→P2, P2→P1), rather than training a joint fusion model on incompatible feature spaces and label schemes.
- Rationale for this simplification (already discussed and agreed): a full fusion architecture combining a CNN-based ECG classifier with a tabular vitals classifier, with no shared synthetic ground truth linking them, was assessed as high-risk for an 8-day build window — particularly because AD8232 signal quality from electrodes during a live demo is unreliable, and a trained model fed noisy/garbage input may confidently output a wrong class with no graceful degradation. A rule-based escalation flag degrades gracefully if ECG signal is bad (it just doesn't trigger; system still works).

### 3. Respiratory rate dataset (PhysioNet) — DROPPED
Was under consideration, group decided to drop the respiration sensor and this dataset entirely. Not in scope. Do not build for it.

## Existing Preprocessing Work (done by a teammate, target fix already applied)

A teammate sent an updated `Preprocess_MLIoT_Dataset.ipynb` (19 cells) alongside fresh copies of the source data (`heartbeat.zip`, `patients_data_with_alerts.xlsx.zip`) — verified byte-identical to what was already in `data/`, so no data drift to worry about.

**ECG Pipeline (cells 0-10):** loads MIT-BIH and PTBDB, reshapes to `(-1, 187, 1)` for 1D CNN input, stratified train/val/test splits, computes balanced class weights (chosen deliberately over blind SMOTE given the larger sample size — reasonable choice, keep as-is). No CNN actually trained yet — preprocessing only.

**IoMT Pipeline (cells 12-17):** loads the Excel file, and the target-column problem flagged in an earlier version of this doc is **already fixed** — the teammate independently wrote their own triage-label function (`assign_triage`, cell 13) rather than using `Predicted Disease`. Drops `Predicted Disease`, `Data Accuracy (%)`, and all four `*Alert` columns from the feature set (correct). Missing value handling, one-hot encoding, label encoding, split-then-scale ordering, and SMOTE-if-imbalanced are all structurally fine and reusable.

**Still open (not a labelling problem, a feature-availability problem):** the feature set `X` still includes `Systolic/Diastolic Blood Pressure` and `Fall Detection` — columns with **no corresponding sensor on the actual ESP32 build**. The label itself is allowed to use them (it's derived from the rich dataset), but a classifier trained on them will degrade at real inference time when those fields are missing/imputed. Not yet resolved — see "What Needs To Be Built" step 1b below.

## Triage Labelling Logic (the core deliverable for this scope)

This is clinically-informed synthetic label generation — same category of approach used in the reference Kaggle notebook (vaibhavsatish/iomt-data) on this same dataset. It is necessary because no real severity ground truth exists in the IoMT dataset.

**Superseded:** an earlier version of this doc specified a `classify_triage()` function with its own HR/SpO2/temp/BP cutoffs. The teammate who sent the updated notebook wrote an independent version (`assign_triage`, cell 13) with different numeric cutoffs. Cross-checked against direct input from a medical student friend (WhatsApp, 2026-06-20, Section 4 group) plus the standard AHA blood-pressure staging chart (anesthguide.com/topic/triage-and-retts), **the teammate's numbers are the better-supported ones for HR, BP, and fall** — so this is now the canonical version, not the original spec above.

```python
def assign_triage(row):
    """
    Returns 'Emergency' / 'Urgent' / 'Non-Urgent', i.e. the MOH P1 / P2 / P3
    tiers under their plain-English names. Checked in severity order
    (Emergency first) — ANY single vital breaching a more severe threshold
    escalates the whole patient, even if other vitals look fine.
    """
    heart_rate = row["Heart Rate (bpm)"]
    spo2 = row["SpO2 Level (%)"]
    systolic = row["Systolic Blood Pressure (mmHg)"]
    diastolic = row["Diastolic Blood Pressure (mmHg)"]
    temperature = row["Body Temperature (°C)"]
    fall = str(row["Fall Detection"]).strip().lower() == "yes"

    # Emergency (P1, RETTS red+orange equivalent)
    if (
        spo2 < 90 or
        heart_rate < 40 or heart_rate >= 130 or
        systolic < 90 or systolic >= 180 or
        diastolic >= 120 or
        temperature < 35 or temperature >= 39.5
    ):
        return "Emergency"

    # Urgent (P2, RETTS yellow equivalent)
    elif (
        spo2 < 94 or
        heart_rate < 50 or heart_rate >= 110 or
        systolic < 100 or systolic >= 140 or
        diastolic >= 90 or
        temperature < 36 or temperature >= 38 or
        fall
    ):
        return "Urgent"

    # Non-Urgent (P3, RETTS green+blue equivalent)
    else:
        return "Non-Urgent"
```

**Validation status, cutoff by cutoff — be precise about this in Q&A, don't claim more than is actually checked:**

| Vital | Cutoff used | Source / validation |
|---|---|---|
| HR, Emergency bound (<40 or ≥130) | exact match to medical student's stated range | confirmed |
| Systolic/diastolic BP, Emergency (≥180 / ≥120) | matches "hypertensive crisis" on both the chart and the friend's message | confirmed |
| Systolic/diastolic BP, Urgent (≥140 / ≥90) | matches the AHA chart's "Stage 2" hypertension boundary | confirmed |
| Fall detected → Urgent (not Emergency) | exact match to friend's statement | confirmed |
| HR, Urgent bound (<50 or ≥110) | interpolated between the friend's normal range and his single outer boundary | reasonable, but not literally stated by the source — own judgment call |
| Low/hypotensive BP cutoff (systolic <90, Emergency) | friend explicitly said low-BP cutoffs are "boleh bincang lagi" (debatable) | **flagged by the source itself as soft** — have a clinical-literature backup ready if pressed |
| SpO2 (90 / 94) and temperature (35/39.5, 36/38) cutoffs | general clinical convention only | **not checked against any source in this exchange** — be upfront about this rather than implying it's vetted |

**Known limitation not addressed by current build:** the friend also noted HR norms are age-dependent (paediatric normal range 130-150 bpm). The dataset and model are adult-vitals-only with no age field — fine to state as an explicit scope limitation if asked, not something to fix today.

## Data Quality (GIGO Check) — Not Garbage, But Not Fully Realistic Either

Checked directly against the real value ranges in `patients_data_with_alerts.xlsx` (see `final-project/visualize_iomt_thresholds.py` for the diagnostic script and saved figures in `output/`). The dataset isn't corrupted — internally consistent, real expert-style structure — but several `assign_triage()` conditions can **never fire** given the actual per-column ranges (HR 60-149, SpO2 80-99, systolic 100-179, diastolic 60-99, temp 36.0-38.0):

- Emergency tier: only `spo2<90` (50.0% of all rows) and `hr>=130` (22.1%) ever trigger. `hr<40`, `systolic<90`, `systolic>=180`, `diastolic>=120`, `temp<35`, `temp>=39.5` fire **0 times** across all 50,000 rows.
- Urgent tier (among non-Emergency rows): `systolic>=140` (50.4%), `fall` (49.8%), `spo2<94` (40.2%), `hr>=110` (28.6%), `diastolic>=90` (24.8%) all fire. `hr<50`, `systolic<100`, `temp<36`, `temp>=38` fire **0 times**.
- Net effect: **temperature contributes nothing to the label at all**, in either tier, in this entire dataset. The raw vitals look like independently-sampled bounded values per column, not a simulation of realistic correlated patient physiology.

**Implication for the trained classifier:** dropping BP/Fall (step 1b) costs nothing on the Emergency tier (never depended on them anyway) but does cost real Urgent/Non-Urgent accuracy, since BP and Fall together explained ~half of all Urgent labels. The classifier's near-perfect Emergency accuracy is not impressive on its own — it reflects a near-deterministic label-feature relationship in synthetic data, not proven generalization to noisy real sensor readings. Frame results this way in Q&A rather than presenting raw accuracy numbers uncritically.

## What Needs To Be Built (in order)

1. ~~Fix the IoMT preprocessing notebook's target.~~ **Done** — see "Existing Preprocessing Work" and "Triage Labelling Logic" above. `assign_triage()` is in place, validated as described, and the drop list is correct.

1b. ~~Decide how to handle BP and Fall Detection as model features.~~ **Done — option (a).** Dropped both from `X`; the classifier trains on HR/SpO2/temp only. They stay in `assign_triage()` for label generation. Implemented in `final-project/train_triage_classifier.py` (a standalone script, not a notebook edit — the notebook is a teammate's shared artifact).

2. **Check class balance on `Triage`.** Already done — Non-Urgent is the minority class (1,520 of 50,000 rows pre-split, ~3%), imbalance ratio ~20x, SMOTE applied on the training split only (after the split, after scaling — correct order, no leakage).

3. ~~Train the primary triage classifier.~~ **Done.** `RandomForestClassifier` (chosen over XGBoost — not installed, RF needs zero new deps and is plenty for 3 features), trained in `final-project/train_triage_classifier.py`. Artifacts saved to `final-project/output/`: `triage_rf_model.joblib`, `triage_scaler.joblib`, `triage_label_mapping.joblib`.
   - Overall accuracy 93.8% vs a 61.0% majority-class baseline.
   - Emergency: perfect (1.00/1.00) — fully determined by SpO2/HR, unaffected by dropping BP/Fall.
   - Urgent: 0.94/0.89.
   - Non-Urgent: 0.18/0.30 — the real cost of dropping BP/Fall (they explained ~half of Urgent classifications per the GIGO analysis below).
   - **Safety-relevant number for Q&A:** of 3,594 truly-Urgent test cases, 410 (11.4%) are misclassified as Non-Urgent — the dangerous under-triage direction. 214 Non-Urgent cases are over-triaged to Urgent — a false alarm, not a missed case.
   - Feature importances: SpO2 0.55, HR 0.34, Temp 0.12 (temp's small nonzero importance is likely noise — its actual threshold conditions never fire in this dataset at all, see GIGO section below).

4. ~~Train or finalize the ECG classifier.~~ **Done.** Confirmed PTBDB binary (normal/abnormal) over MIT-BIH's 5-class scheme — the MIT-BIH preprocessing in the notebook stays unused, this model's only job is the escalation flag. TensorFlow/Keras added as a new dependency (chosen over PyTorch — native `class_weight` support in `model.fit()`, fewer lines under time pressure); trained in standalone `final-project/train_ecg_classifier.py` (not a notebook edit). Small 1D CNN: 2x (Conv1D + MaxPooling1D) → GlobalAveragePooling1D → Dense + Dropout → sigmoid, trained with class weights + early stopping (stopped at 30 epochs, ~30s total on CPU, no GPU needed for this size).
   - **Reproducibility bug found and fixed (via an independent Opus audit, 2026-06-25):** the script seeded the train/test split but not Keras itself (weight init, dropout, batch shuffling) — violates this project's own "seed everything" rule. Re-running produced wildly different operating-point metrics run to run (abnormal-class false-negative rate ranged ~4-16% across 3 runs) even though AUC stayed stable (~0.975). Fixed with one line: `keras.utils.set_random_seed(RANDOM_STATE)` at the top of `main()`. Verified deterministic across 2 repeat runs after the fix.
   - **Current canonical numbers (reproducible, confirmed twice post-fix):** Test AUC 0.975, overall accuracy 92%. Normal: 0.82 precision / 0.94 recall. Abnormal: 0.97 precision / 0.92 recall.
   - Confusion matrix: 168 of 2,102 truly-abnormal cases missed (predicted Normal) — **8.0% false-negative rate**, the direction that matters for the escalation flag (a missed abnormal ECG means no escalation when one was warranted). 51 of 809 truly-normal cases over-flagged as Abnormal — the safe direction (unnecessary escalation, not a missed case).
   - **Don't retrain casually before the demo** — re-running is now deterministic given the fix, but there's no upside to touching a working, verified artifact this close to presentation day.
   - Artifacts in `final-project/output/`: `ecg_cnn_model.keras`, confusion matrix, ROC curve.

5. ~~Build the fusion/escalation logic.~~ **Done.** `final-project/triage_fusion.py` (standalone, not a notebook edit) — `escalate_tier()` is a plain rule (`TIER_ORDER = ["Non-Urgent", "Urgent", "Emergency"]`, move up one index if `ecg_abnormal`, capped at Emergency), not a third trained model, exactly as scoped. `fuse_triage_decision()` loads both saved models and ties them together; returns a `TriageDecision` dataclass (`base_tier`, `ecg_abnormal`, `final_tier`) so callers get the full picture, not just the final label.
   - Demo in `main()` runs 4 cases (no escalation, Non-Urgent→Urgent, Urgent→Emergency, capped at Emergency) — all 4 verified correct, and re-verified after a cosmetic fix (below).
   - **Fixed (Opus audit):** `predict_triage_tier()` passed a bare list to the scaler, which was fit on a named DataFrame — caused a `sklearn` "X does not have valid feature names" warning on every call. Harmless to correctness (column order was preserved) but noisy output during a live demo looks bad. Fixed by building a one-row DataFrame with `TRIAGE_FEATURE_COLUMNS` before scaling. Re-verified: same 4/4 correct results, no warnings.
   - **Hit the Non-Urgent weakness from step 3 head-on while building the demo:** synthetic "obviously normal" vitals (HR~60-95, SpO2~96-99, temp~36.5-37.1) were *never* classified as Non-Urgent by the trained RF — it only predicts Non-Urgent for specific real rows pulled from the actual dataset. Confirms the earlier 0.18 precision / 0.30 recall finding wasn't a fluke; the model's Non-Urgent decision boundary is narrow and doesn't generalize to "obviously fine" hand-picked values the way you'd expect. Worth knowing if asked to demo live with made-up "healthy" numbers — they may not actually produce Non-Urgent.

6. **Package for inference on whatever machine hosts Node-RED** (confirm this topology first — see Hardware section above). Needs to accept live sensor readings (HR, SpO2, temp from MAX30102+DS18B20; ECG-derived flag from AD8232 pipeline, read via ESP32's built-in ADC — segmentation/preprocessing of the live analog ECG signal into the 187-sample format the CNN expects is a separate, nontrivial piece of work flagged as a risk area, not yet solved) arriving over MQTT from the ESP32, and output a final P1/P2/P3 decision to be published back to MQTT for Node-RED/Streamlit to consume. Do not assume the ESP32 itself runs any ML inference — it almost certainly can't host this.

## Known Risks / Things To Watch

- **ECG live signal quality.** AD8232 electrode-based ECG is notoriously noisy with non-ideal placement. The model expects clean, segmented, normalized 187-sample heartbeat windows — getting reliably good input live during a demo is the single highest-risk part of this build. Have a fallback (e.g., skip ECG escalation entirely and demo on vitals-only triage) in case live ECG segmentation isn't working reliably by demo day.
- **Topology confirmation needed.** Confirm where Mosquitto/Node-RED/Streamlit/ML inference actually run, now that ESP32 (not Pi) is the sensor node. ESP32 publishes over MQTT; it does not host the broker or run inference.
- **ESP32 ADC quirks.** ESP32's ADC2 pins conflict with WiFi usage and are known to be less reliable; use ADC1 pins for the AD8232 analog read if WiFi is active simultaneously (which it will be, since ESP32 needs WiFi for MQTT).
- **Train-serve feature mismatch.** Don't let any synthetic-dataset-only columns (`Data Accuracy (%)`, `Predicted Disease`, the `*Alert` columns) leak into the trained feature set — they don't exist as real-world sensor outputs and won't be available at inference.
- **Threshold defensibility.** Mostly de-risked now — HR, BP, and fall cutoffs are corroborated by a medical student's direct input plus the AHA BP staging chart (see validation table above). SpO2 and temperature cutoffs are still just general clinical convention, unchecked against any source — don't imply otherwise if asked.

## Style/Workflow Preferences (for whoever continues this in Claude Code)

- Prefers complete, runnable code over fragments.
- Wants clear explanations of what changed and why when code is revised.
- Direct, no padding, no repeating already-resolved points.
- Train/test split must happen before any preprocessing step that calls `.fit()` (e.g. StandardScaler) — this is a known recurring bug source from past coursework, double-check the IoMT pipeline respects this (current notebook does split-then-scale correctly in cell 17 — preserve that order in any edits).
