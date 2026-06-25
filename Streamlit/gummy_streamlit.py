import time
import random
import threading

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
ML_API_URL = "http://localhost:5000/predict"

SENSOR_KEYS = ["sensor1", "sensor2", "sensor3"]
SENSOR_LABELS = {"sensor1": "Sensor 1", "sensor2": "Sensor 2", "sensor3": "Sensor 3"}
DATA_TOPICS = {k: f"data/{k}" for k in SENSOR_KEYS}
CMD_TOPICS = {k: f"cmd/{k}/scan" for k in SENSOR_KEYS}

# Rough simulated value ranges per sensor, just so the mock data looks plausible.
MOCK_RANGES = {"sensor1": (15, 35), "sensor2": (0.5, 3.0), "sensor3": (80, 120)}


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class ScanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.sensors = {
                k: {"count": 0, "readings": [], "last_value": None} for k in SENSOR_KEYS
            }
            self.status = "in_progress"
            self.last_result = None
            self.notification = None

    def add_reading(self, sensor_key, value):
        with self.lock:
            if self.status != "in_progress":
                return
            s = self.sensors[sensor_key]
            if s["count"] < 3:
                s["readings"].append(value)
                s["count"] += 1
            s["last_value"] = value
            self._check_complete_locked()

    def _check_complete_locked(self):
        all_done = all(self.sensors[k]["count"] >= 3 for k in SENSOR_KEYS)
        if all_done and self.status == "in_progress":
            self.status = "processing"
            averages = {k: sum(v["readings"]) / len(v["readings"]) for k, v in self.sensors.items()}
            threading.Thread(target=self._run_ml_call, args=(averages,), daemon=True).start()

    def _run_ml_call(self, averages):
        if st.session_state.get("mock_mode", True):
            result = self._mock_predict(averages)
        else:
            result = self._real_predict(averages)

        with self.lock:
            self.last_result = {**averages, **result, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.status = "complete"
            self.notification = "Scan complete! Check the Results tab."

    def _mock_predict(self, averages):
        time.sleep(1)  # pretend there's some processing delay
        verdict = random.choice(["good", "good", "bad"])  # mostly "good" for demo purposes
        return {"verdict": verdict, "confidence": round(random.uniform(0.7, 0.99), 2)}

    def _real_predict(self, averages):
        import requests
        try:
            resp = requests.post(ML_API_URL, json=averages, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"verdict": "error", "confidence": 0.0, "error": str(e)}

    def snapshot(self):
        with self.lock:
            return {
                "sensors": {k: dict(v) for k, v in self.sensors.items()},
                "status": self.status,
                "last_result": dict(self.last_result) if self.last_result else None,
                "notification": self.notification,
            }

    def clear_notification(self):
        with self.lock:
            self.notification = None


@st.cache_resource
def get_state():
    return ScanState()


@st.cache_resource
def get_mqtt_client(_state: ScanState):
    """Only connects to MQTT when mock mode is off. Imports paho-mqtt lazily
    so the app runs fine for UI testing even if that package isn't installed."""
    import paho.mqtt.client as mqtt

    client = mqtt.Client()

    def on_connect(client, userdata, flags, rc):
        for topic in DATA_TOPICS.values():
            client.subscribe(topic)

    def on_message(client, userdata, msg):
        for key, topic in DATA_TOPICS.items():
            if msg.topic == topic:
                try:
                    value = float(msg.payload.decode())
                except ValueError:
                    continue
                _state.add_reading(key, value)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()
    return client


def trigger_scan(state: ScanState, client, sensor_key: str, mock_mode: bool):
    if mock_mode:
        lo, hi = MOCK_RANGES[sensor_key]
        value = round(random.uniform(lo, hi), 2)
        state.add_reading(sensor_key, value)
    else:
        client.publish(CMD_TOPICS[sensor_key], "1")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sensor Scan Dashboard", layout="wide")

with st.sidebar:
    st.header("⚙️ Dev Settings")
    mock_mode = st.toggle("Mock Mode (no MQTT/API needed)", value=True)
    st.session_state["mock_mode"] = mock_mode
    if mock_mode:
        st.caption("Scan buttons generate random values instantly. ML verdict is randomly simulated.")
    else:
        st.caption("Scan buttons publish real MQTT commands and call the real ML API.")

state = get_state()
client = get_mqtt_client(state) if not mock_mode else None

st_autorefresh(interval=1000, key="refresh")

tab_scan, tab_results = st.tabs(["📡 Scan", "📋 Results"])

with tab_scan:
    snap = state.snapshot()

    if snap["notification"]:
        st.toast(snap["notification"], icon="✅")
        state.clear_notification()

    top_l, top_r = st.columns([3, 1])
    with top_l:
        status_map = {
            "in_progress": "🟡 In progress",
            "processing": "🔵 Processing (calling ML model)...",
            "complete": "🟢 Complete",
        }
        st.subheader(status_map.get(snap["status"], snap["status"]))
    with top_r:
        if st.button("🔄 Start New Scan", type="primary", use_container_width=True):
            state.reset()
            st.rerun()

    st.divider()
    cols = st.columns(3)

    for col, key in zip(cols, SENSOR_KEYS):
        with col:
            s = snap["sensors"][key]
            st.markdown(f"#### {SENSOR_LABELS[key]}")

            value = s["last_value"]
            st.metric("Current Value", f"{value:.2f}" if value is not None else "—")
            st.progress(s["count"] / 3, text=f"{s['count']}/3 scans")

            disabled = s["count"] >= 3 or snap["status"] != "in_progress"
            if st.button(f"Scan {SENSOR_LABELS[key]}", key=f"btn_{key}", disabled=disabled, use_container_width=True):
                trigger_scan(state, client, key, mock_mode)
                if mock_mode:
                    st.rerun()
                else:
                    st.toast(f"Scan command sent to {SENSOR_LABELS[key]}...")

with tab_results:
    snap = state.snapshot()
    result = snap["last_result"]

    if not result:
        st.info("No scan completed yet. Go to the Scan tab and run all 3 sensors.")
    else:
        st.subheader("Last Scan Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensor 1 (avg)", f"{result['sensor1']:.2f}")
        c2.metric("Sensor 2 (avg)", f"{result['sensor2']:.2f}")
        c3.metric("Sensor 3 (avg)", f"{result['sensor3']:.2f}")

        verdict = result.get("verdict", "unknown")
        color = "green" if verdict == "good" else "red"
        st.markdown(f"### Verdict: :{color}[{verdict.upper()}]")
        st.write(f"**Confidence:** {result.get('confidence', 0):.2f}")
        st.caption(f"Scanned at {result['timestamp']}")

        if verdict == "error":
            st.error(f"ML API call failed: {result.get('error', 'unknown error')}")
