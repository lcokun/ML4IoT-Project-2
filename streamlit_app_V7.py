"""Streamlit dashboard for the IoMT triage system.

    # terminal 1 - broker
    mosquitto -v

    # terminal 2 - subscriber, writes latest_data.json
    python mqtt_reciever.py

    # terminal 3 - data source
    # have the ESP32 powered on and connected to the same WiFi network
    # (or a simulator/mosquitto_pub test message, see chat for examples)

    # terminal 4 - dashboard
    streamlit run streamlit_app.py

This app is read-only against latest_data.json -- it never publishes MQTT
messages itself. mqtt_reciever.py only forwards raw vitals; it doesn't run
the RF/CNN triage models. THIS file calls triage_engine.py directly on each
new reading (see run_triage() below) so the tier shown is always computed
from the live data, not waiting on the receiver to do it.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import json

from triage_engine import TriageEngine, TIER_COLORS, TIER_ORDER, WAVEFORM_LENGTH

st.set_page_config(page_title="IoMT Triage Monitor", page_icon="🩺", layout="wide")

LATEST_DATA_PATH = "latest_data.json"  # must match DATA_FILE in mqtt_reciever.py
REFRESH_INTERVAL_S = 2
HISTORY_WINDOW = 60  # how many past readings to show in trend charts

# How often the RF+CNN models actually re-run. Kept separate from
# REFRESH_INTERVAL_S: the gauges/vitals on screen still update every 2s,
# but re-running the CNN that often is wasteful when nothing's changed.
# The cached tier is reused in between -- see maybe_run_triage() below.
INFERENCE_INTERVAL_S = 10


@st.cache_resource(show_spinner=False)
def get_engine() -> TriageEngine:
    """Load the RF + CNN model artifacts once per Streamlit server process,
    not on every 2-second rerun -- loading a Keras model from disk that
    often is not cheap."""
    from triage_engine import engine
    engine.load()
    return engine


def read_sensor_data() -> dict | None:
    """Read the receiver's output file. Returns None if it doesn't exist yet
    (e.g. mqtt_reciever.py hasn't received a first message), rather than
    crashing the whole app."""
    try:
        with open(LATEST_DATA_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def normalize_reading(raw: dict | None) -> dict | None:
    """Translate mqtt_reciever.py's field names into the names the rest of
    this dashboard expects. This only maps *inputs* (vitals + raw ECG
    buffer) -- triage *outputs* (final_tier etc.) are computed separately
    by run_triage() below, since the receiver never produces them."""
    if raw is None:
        return None
    return {
        "hr": raw.get("bpm"),
        "spo2": raw.get("spo2"),
        "temp": raw.get("temperature"),
        "ecg_window": raw.get("ecg_window") or [],
        "received_at": raw.get("received_at"),
        "timestamp": raw.get("timestamp"),
    }


def build_ecg_segment(ecg_window: list, length: int = WAVEFORM_LENGTH):
    """Turn the receiver's rolling raw-ECG buffer into a fixed-length,
    min-max normalized segment the CNN expects.

    Always returns (segment, is_full_window) -- segment is never None, so
    the model can always run immediately, even before any real ECG data
    has arrived (flat 0.5 placeholder) or while the buffer is still
    filling (left-padded with the earliest sample). is_full=False in
    either case flags the result as provisional, not clinically
    meaningful, rather than blocking inference outright.
    """
    if not ecg_window:
        return np.full(length, 0.5), False

    arr = np.asarray(ecg_window, dtype=float)
    is_full = len(arr) >= length

    if is_full:
        segment = arr[-length:]
    else:
        pad = np.full(length - len(arr), arr[0])
        segment = np.concatenate([pad, arr])

    if segment.max() == segment.min():
        # Flat signal (e.g. leads-off, or no real samples yet) -- can't
        # min-max normalize a constant array. Feed the model a neutral
        # mid-scale flat line instead of refusing to run at all; ecg_abnormal
        # from a flat input isn't meaningful, callers should treat it as
        # provisional via is_full=False.
        return np.full(length, 0.5), is_full

    normalized = (segment - segment.min()) / (segment.max() - segment.min())
    return normalized, is_full


def generate_synthetic_ecg(pattern: str, length: int = 200) -> list[float]:
    """Build a fake raw ECG buffer for manual test mode, in the same rough
    value range as real AD8232 ADC samples (~0-4095) so it flows through
    build_ecg_segment() exactly like live data would. This is a crude
    pulse-train shape for functional testing only -- NOT realistic ECG
    morphology, and not meant to resemble any real diagnostic waveform.
    """
    t = np.linspace(0, 1, length)
    baseline = 2000.0

    if pattern == "Flat / no signal":
        return [baseline] * length

    bpm = 75
    beat_interval = 60.0 / bpm
    signal = np.full(length, baseline)
    for beat_time in np.arange(0, 1, beat_interval):
        spike = 1800 * np.exp(-((t - beat_time) ** 2) / (2 * 0.01 ** 2))
        signal += spike

    if pattern == "Abnormal (irregular)":
        rng = np.random.default_rng()
        signal += rng.normal(0, 300, length)
        # one extra out-of-rhythm spike to mimic an irregular beat
        signal += 1500 * np.exp(-((t - 0.5) ** 2) / (2 * 0.005 ** 2))

    return signal.tolist()


# Neutral placeholder vitals -- NOT a clinical claim, just a stand-in so the
# pipeline can run end-to-end before every single sensor has reported in at
# least once. Any tier computed using these is flagged via
# result["vitals_estimated"] so the UI can show it's provisional.
DEFAULT_VITALS = {"hr": 75, "spo2": 98, "temp": 36.8}


def run_triage(reading: dict, engine: TriageEngine) -> dict:
    """Compute final_tier/base_tier/ecg_abnormal/ecg_probability for a
    normalized reading by calling the loaded RF+CNN models directly.

    Runs immediately on whatever is currently available rather than
    waiting for every sensor to have reported at least once -- any vital
    or ECG signal still missing is filled with a neutral placeholder and
    listed in result["vitals_estimated"] / result["ecg_window_full"] so the
    UI can flag it as provisional instead of either blocking entirely or
    silently presenting a guess as real.

    Always returns a dict with final_tier/base_tier/etc. present (None
    only if the model itself isn't loaded), plus 'triage_error' so the UI
    can explain *why* a tier isn't showing instead of just looking broken.
    """
    result = dict(reading)
    result.update(
        final_tier=None,
        base_tier=None,
        ecg_abnormal=None,
        ecg_probability=None,
        triage_error=None,
        ecg_window_full=False,
        vitals_estimated=[],
    )

    if not engine.is_ready and not engine.load():
        result["triage_error"] = engine.load_error
        return result

    hr, spo2, temp = reading.get("hr"), reading.get("spo2"), reading.get("temp")
    estimated = []
    if hr is None:
        hr, estimated = DEFAULT_VITALS["hr"], estimated + ["hr"]
    if spo2 is None:
        spo2, estimated = DEFAULT_VITALS["spo2"], estimated + ["spo2"]
    if temp is None:
        temp, estimated = DEFAULT_VITALS["temp"], estimated + ["temp"]
    result["vitals_estimated"] = estimated

    segment, is_full = build_ecg_segment(reading.get("ecg_window"))
    result["ecg_window_full"] = is_full

    try:
        triage = engine.evaluate(hr, spo2, temp, segment)
    except Exception as exc:  # noqa: BLE001 - surface to the UI, don't crash the app
        result["triage_error"] = f"Inference failed: {exc}"
        return result

    result.update(
        final_tier=triage.final_tier,
        base_tier=triage.base_tier,
        ecg_abnormal=triage.ecg_abnormal,
        ecg_probability=triage.ecg_probability,
    )
    return result


def maybe_run_triage(reading: dict, engine: TriageEngine) -> dict:
    """Wraps run_triage() with a cache so the RF+CNN models only actually
    re-run at most once every INFERENCE_INTERVAL_S seconds, not on every
    2-second UI refresh. Vitals/ECG display still update every refresh --
    only the (expensive) model call itself is throttled.

    Forces a fresh inference immediately the first time the ECG buffer
    becomes "full" (not just padded), so the first real result isn't stuck
    waiting out the throttle window with a provisional/padded one.
    """
    cache = st.session_state.get("triage_cache")
    now = time.time()

    is_new_data = cache is None or cache.get("source_received_at") != reading.get("received_at")
    throttle_elapsed = cache is None or (now - cache["computed_at"]) >= INFERENCE_INTERVAL_S
    ecg_just_completed = (
        cache is not None
        and not cache["result"].get("ecg_window_full")
        and (reading.get("ecg_window") or [])
        and len(reading["ecg_window"]) >= WAVEFORM_LENGTH
    )

    if cache is None or (is_new_data and (throttle_elapsed or ecg_just_completed)):
        result = run_triage(reading, engine)
        st.session_state.triage_cache = {
            "computed_at": now,
            "source_received_at": reading.get("received_at"),
            "result": result,
        }
        return result

    # Reuse the cached triage outputs, but keep showing the *live* vitals/
    # ECG buffer from `reading` rather than the stale snapshot they were
    # computed from -- only the tier/probability fields are cached.
    merged = dict(reading)
    cached_fields = (
        "final_tier", "base_tier", "ecg_abnormal",
        "ecg_probability", "triage_error", "ecg_window_full",
    )
    for key in cached_fields:
        merged[key] = cache["result"].get(key)
    return merged


# ---------------------------------------------------------------------------
# Minimal, restrained styling: this is a clinical readout, not a marketing
# page, so color is reserved for one job -- signaling triage severity --
# rather than decoration. Everything else stays neutral.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        .tier-banner {
            border-radius: 10px;
            padding: 1.1rem 1.5rem;
            color: white;
            font-size: 1.4rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-align: center;
            margin-bottom: 0.75rem;
        }
        .tier-sub {
            font-size: 0.95rem;
            font-weight: 400;
            opacity: 0.9;
        }
        .pipeline-step {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.95rem;
            padding: 0.35rem 0;
        }
        .pipeline-dot {
            width: 10px; height: 10px; border-radius: 50%;
            display: inline-block;
        }
        .stale-warning {
            background: #fff3cd; border: 1px solid #ffe69c;
            border-radius: 6px; padding: 0.6rem 1rem; font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def gauge_figure(value: float, title: str, value_range: tuple[float, float],
                  bands: list[tuple[float, float, str]], unit: str) -> go.Figure:
    """A speedometer-style gauge with colored severity bands."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": f" {unit}", "font": {"size": 30}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": list(value_range)},
                "bar": {"color": "#37474f", "thickness": 0.25},
                "steps": [{"range": [b[0], b[1]], "color": b[2]} for b in bands],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_pipeline_progress(vitals_received: bool, ecg_window_ready: bool, triage_done: bool) -> None:
    steps = [
        ("Collecting vitals from ESP32 (HR / SpO2 / Temp)", vitals_received),
        ("Buffering ECG waveform (187-sample window)", ecg_window_ready),
        ("Running triage model + ECG escalation check", triage_done),
    ]
    cols = st.columns(len(steps))
    for col, (label, done) in zip(cols, steps):
        color = "#2e7d32" if done else "#b0bec5"
        icon = "✓" if done else "…"
        col.markdown(
            f'<div class="pipeline-step">'
            f'<span class="pipeline-dot" style="background:{color}"></span>'
            f'<span>{icon} {label}</span></div>',
            unsafe_allow_html=True,
        )


def render_tier_banner(final_tier: str | None, base_tier: str | None, ecg_abnormal: bool | None) -> None:
    if final_tier is None:
        st.markdown(
            '<div class="tier-banner" style="background:#90a4ae;">Waiting for first reading…</div>',
            unsafe_allow_html=True,
        )
        return

    color = TIER_COLORS.get(final_tier, "#607d8b")
    escalated = base_tier is not None and final_tier != base_tier
    sub = ""
    if escalated:
        sub = f'<div class="tier-sub">Escalated from {base_tier} due to abnormal ECG</div>'
    elif ecg_abnormal:
        sub = '<div class="tier-sub">ECG abnormal, but already at top tier</div>'

    st.markdown(
        f'<div class="tier-banner" style="background:{color};">{final_tier}{sub}</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every=REFRESH_INTERVAL_S)
def live_dashboard() -> None:
    """Everything that depends on the latest reading lives in this
    fragment. Streamlit reruns *only* this function every
    REFRESH_INTERVAL_S seconds -- the page title/layout above it, and the
    user's scroll position, are untouched. This replaces the previous
    time.sleep() + st.rerun() pattern, which reran (and visibly redrew)
    the entire page on every cycle.

    Data source is either the live MQTT-fed file, or a manual test
    reading set via the sidebar (see render_manual_input_sidebar() in
    main()) -- toggled by st.session_state.data_mode.
    """
    engine = get_engine()

    is_manual = st.session_state.get("data_mode") == "Manual test input"
    if is_manual:
        latest = st.session_state.get("manual_reading")
    else:
        raw = read_sensor_data()
        latest = normalize_reading(raw)

    if latest is not None:
        latest = maybe_run_triage(latest, engine)

    # --- Build trend history in-session ------------------------------------
    # mqtt_reciever.py only ever writes the *latest* reading, not a log, so
    # the dashboard accumulates its own short history across reruns instead.
    # Manual test readings go in a separate history so they never mix into
    # the live trend chart (and vice versa) when switching modes.
    history_key = "manual_history" if is_manual else "vitals_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    if latest is not None and latest.get("received_at") is not None:
        history = st.session_state[history_key]
        last_seen = history[-1]["received_at"] if history else None
        if latest["received_at"] != last_seen:
            history.append(latest)
            st.session_state[history_key] = history[-HISTORY_WINDOW:]

    vitals_history = st.session_state[history_key]
    latest_ecg = {"samples": latest["ecg_window"]} if latest and latest.get("ecg_window") else None

    # --- No data at all yet -------------------------------------------------
    if latest is None:
        if st.session_state.get("data_mode") == "Manual test input":
            st.info("Set vitals/ECG pattern in the sidebar, then click **Apply manual reading**.")
        else:
            st.info(
                f"Waiting for `{LATEST_DATA_PATH}` -- make sure mosquitto, "
                "mqtt_reciever.py, and the ESP32 (or simulator) are all running."
            )
    elif latest.get("triage_error"):
        st.warning(f"Triage: {latest['triage_error']}")

    # --- Pipeline progress -------------------------------------------------
    vitals_received = latest is not None
    ecg_window_ready = latest is not None and latest.get("ecg_window_full", False)
    triage_done = latest is not None and latest.get("final_tier") is not None
    render_pipeline_progress(vitals_received, ecg_window_ready, triage_done)
    if latest is not None and not ecg_window_ready and latest.get("ecg_window"):
        st.caption(
            f"Buffering ECG: {len(latest['ecg_window'])}/{WAVEFORM_LENGTH} samples "
            "-- tier shown below uses a padded/provisional window until full."
        )
    st.divider()

    # --- Staleness check ----------------------------------------------------
    if not is_manual and latest is not None and latest.get("received_at") is not None:
        age_s = time.time() - latest["received_at"]
        if age_s > 15:
            st.markdown(
                f'<div class="stale-warning">⚠️ Last reading was {age_s:.0f}s ago. '
                "Check that the ESP32 (or simulator) and mqtt_reciever.py are running.</div>",
                unsafe_allow_html=True,
            )
            st.write("")

    left, right = st.columns([1, 1.3])

    # --- Left: triage result + gauges --------------------------------------
    with left:
        st.subheader("Triage result")
        render_tier_banner(
            latest.get("final_tier") if latest else None,
            latest.get("base_tier") if latest else None,
            latest.get("ecg_abnormal") if latest else None,
        )
        if latest is not None and latest.get("ecg_probability") is not None:
            st.caption(f"ECG abnormal-probability score: {latest['ecg_probability']:.2f}")

        st.subheader("Vitals")
        estimated = latest.get("vitals_estimated") if latest else None
        if estimated:
            st.caption(
                f"⚠️ Placeholder value(s) used (sensor hasn't reported yet): {', '.join(estimated)}"
            )
        gcols = st.columns(3)
        hr = latest.get("hr") if latest else None
        spo2 = latest.get("spo2") if latest else None
        temp = latest.get("temp") if latest else None

        with gcols[0]:
            st.plotly_chart(
                gauge_figure(
                    hr or 0, "Heart Rate", (0, 200),
                    [(0, 40, "#c62828"), (40, 50, "#f9a825"), (50, 110, "#2e7d32"),
                     (110, 130, "#f9a825"), (130, 200, "#c62828")],
                    "bpm",
                ),
                use_container_width=True,
            )
        with gcols[1]:
            st.plotly_chart(
                gauge_figure(
                    spo2 or 0, "SpO2", (0, 100),
                    [(0, 90, "#c62828"), (90, 94, "#f9a825"), (94, 100, "#2e7d32")],
                    "%",
                ),
                use_container_width=True,
            )
        with gcols[2]:
            st.plotly_chart(
                gauge_figure(
                    temp or 0, "Body Temp", (30, 42),
                    [(30, 35, "#c62828"), (35, 36, "#f9a825"), (36, 38, "#2e7d32"),
                     (38, 39.5, "#f9a825"), (39.5, 42, "#c62828")],
                    "°C",
                ),
                use_container_width=True,
            )

    # --- Right: ECG waveform + trend ---------------------------------------
    with right:
        st.subheader("ECG waveform (latest window)")
        if latest_ecg is not None:
            samples = latest_ecg["samples"]
            fig = go.Figure(go.Scatter(y=samples, mode="lines", line=dict(color="#37474f", width=1.5)))
            fig.update_layout(
                height=220, margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="sample (187 per window, ~200Hz)", yaxis_title="amplitude",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ECG window received yet.")

        st.subheader("Recent trend")
        if vitals_history:
            df = pd.DataFrame(vitals_history)
            if "received_at" in df.columns:
                df["time"] = pd.to_datetime(df["received_at"], unit="s", errors="coerce")
            fig = go.Figure()
            for col, color in [("hr", "#1565c0"), ("spo2", "#2e7d32"), ("temp", "#ef6c00")]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df.get("time", df.index), y=df[col], name=col, mode="lines+markers"))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=20), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No history yet.")

    # --- History table -------------------------------------------------------
    with st.expander(f"Full reading history"):
        if vitals_history:
            # Drop the raw ECG sample buffer -- a list-per-cell renders badly
            # in a table and isn't useful here; the waveform panel above
            # already shows it.
            table_rows = [
                {k: v for k, v in row.items() if k != "ecg_window"}
                for row in reversed(vitals_history)
            ]
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("Nothing recorded yet.")

    st.caption(
        f"Vitals refresh every {REFRESH_INTERVAL_S}s · "
        f"Triage model re-runs at most every {INFERENCE_INTERVAL_S}s"
    )


def render_data_source_sidebar() -> None:
    """Lets the user switch between the live MQTT-fed file and a manual
    test reading they enter by hand -- useful for checking the RF/CNN
    pipeline works correctly without needing the ESP32 connected at all.
    """
    st.sidebar.header("Data source")
    st.sidebar.radio(
        "Mode",
        ["Live (MQTT)", "Manual test input"],
        key="data_mode",
        help="Manual mode lets you type in vitals/ECG values directly to "
             "test the triage model without the ESP32 connected.",
    )

    if st.session_state.get("data_mode") != "Manual test input":
        return

    st.sidebar.subheader("Manual vitals")
    m_hr = st.sidebar.number_input("Heart rate (bpm)", 0, 250, 75)
    m_spo2 = st.sidebar.number_input("SpO2 (%)", 0, 100, 98)
    m_temp = st.sidebar.number_input("Temperature (°C)", 30.0, 42.0, 36.8, step=0.1)

    st.sidebar.subheader("Manual ECG")
    ecg_pattern = st.sidebar.selectbox(
        "ECG pattern",
        ["Normal sinus", "Abnormal (irregular)", "Flat / no signal"],
        help="Synthetic test waveform, not real ECG morphology -- for "
             "checking the pipeline runs and responds, not for clinical accuracy.",
    )

    if st.sidebar.button("Apply manual reading", type="primary"):
        st.session_state.manual_reading = {
            "hr": m_hr,
            "spo2": m_spo2,
            "temp": m_temp,
            "ecg_window": generate_synthetic_ecg(ecg_pattern),
            "received_at": time.time(),
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        }
        st.session_state.triage_cache = None  # force a fresh inference on this new test reading


def main() -> None:
    st.title("🩺 IoMT Triage Monitor")
    st.caption("ESP32 (MAX30102 + DS18B20 + AD8232) → MQTT → triage model fusion")
    render_data_source_sidebar()
    live_dashboard()


if __name__ == "__main__":
    main()


