"""Streamlit dashboard for the IoMT triage system.

Run alongside mqtt_listener.py (which owns the MQTT connection and writes
to storage/*.json) and either simulator.py or the real ESP32 firmware:

    # terminal 1
    mosquitto -c mosquitto.conf

    # terminal 2
    python mqtt_listener.py

    # terminal 3 (pick one)
    python simulator.py --scenario normal
    # ...or have the real ESP32 powered on and connected to the same network

    # terminal 4
    streamlit run streamlit_app.py

This app is read-only against storage/*.json -- it never publishes MQTT
messages itself, and it doesn't run the triage model directly (the
listener does, on each completed reading, so results stay consistent
regardless of which client is viewing the dashboard).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import json 

from triage_engine import TIER_COLORS, TIER_ORDER

LATEST_DATA_PATH = "esp32"/"latest_data.json"
REFRESH_INTERVAL_S = 2
HISTORY_WINDOW = 60  # how many past readings to show in trend charts

def read_sensor_data() -> dict:
    with open(LATEST_DATA_PATH) as f:
        return json.load(f)

st.set_page_config(page_title="IoMT Triage Monitor", page_icon="🩺", layout="wide")

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


def main() -> None:
    st.title("🩺 IoMT Triage Monitor")
    st.caption("ESP32 (MAX30102 + DS18B20 + AD8232) → MQTT → triage model fusion")

    latest = read_sensor_data.latest_vitals()
    latest_ecg = read_sensor_data.latest_ecg_window()
    vitals_history = read_sensor_data.read_vitals(limit=HISTORY_WINDOW)

    # --- Pipeline progress -------------------------------------------------
    vitals_received = latest is not None
    ecg_window_ready = latest_ecg is not None
    triage_done = latest is not None and "final_tier" in latest
    render_pipeline_progress(vitals_received, ecg_window_ready, triage_done)
    st.divider()

    # --- Staleness check ----------------------------------------------------
    if latest is not None:
        age_s = time.time() - latest.get("received_at", time.time())
        if age_s > 15:
            st.markdown(
                f'<div class="stale-warning">⚠️ Last reading was {age_s:.0f}s ago. '
                "Check that the ESP32 (or simulator) and mqtt_listener.py are running.</div>",
                unsafe_allow_html=True,
            )
            st.write("")

    left, right = st.columns([1, 1.3])

    # --- Left: triage result + gauges --------------------------------------
    with left:
        st.subheader("Triage result")
        if latest is not None and "triage_error" in latest:
            st.error(f"Model error on last reading: {latest['triage_error']}")
        render_tier_banner(
            latest.get("final_tier") if latest else None,
            latest.get("base_tier") if latest else None,
            latest.get("ecg_abnormal") if latest else None,
        )
        if latest is not None and "ecg_probability" in latest:
            st.caption(f"ECG abnormal-probability score: {latest['ecg_probability']:.2f}")

        st.subheader("Vitals")
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
            if "timestamp" in df.columns:
                df["time"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            fig = go.Figure()
            for col, color in [("hr", "#1565c0"), ("spo2", "#2e7d32"), ("temp", "#ef6c00")]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df.get("time", df.index), y=df[col], name=col, mode="lines+markers"))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=20), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No history yet.")

    # --- History table -------------------------------------------------------
    with st.expander(f"Full reading history ({read_sensor_data.count_vitals()} total stored)"):
        if vitals_history:
            df = pd.DataFrame(list(reversed(vitals_history)))
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("Nothing recorded yet.")

    st.caption(
        f"Auto-refreshing every {REFRESH_INTERVAL_S}s · "
        f"{json_store.count_vitals()} vitals readings · {read_sensor_data.count_ecg_windows()} ECG windows stored"
    )
    time.sleep(REFRESH_INTERVAL_S)
    st.rerun()


if __name__ == "__main__":
    main()

