import time
import threading

import requests
import streamlit as st
import paho.mqtt.client as mqtt
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# CONFIG - edit these to match your setup
# ---------------------------------------------------------------------------
#MQTT_BROKER = "localhost"      # <-- change to your broker IP
#MQTT_PORT = 1883
#ML_API_URL = "http://localhost:5000/predict"  # <-- change to your Python ML API

SENSOR_KEYS = ["sensor1", "sensor2", "sensor3"]
SENSOR_LABELS = {"sensor1": "Sensor 1", "sensor2": "Sensor 2", "sensor3": "Sensor 3"}
DATA_TOPICS = {k: f"data/{k}" for k in SENSOR_KEYS}      # ESP32 -> dashboard
CMD_TOPICS = {k: f"cmd/{k}/scan" for k in SENSOR_KEYS}   # dashboard -> ESP32


# ---------------------------------------------------------------------------
# Shared state - lives once per server process (NOT per browser session),
# since the scan session represents one physical device being tested.
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
            self.status = "in_progress"  # in_progress | processing | complete
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
            threading.Thread(target=self._call_ml_api, args=(averages,), daemon=True).start()

    def _call_ml_api(self, averages):
        try:
            resp = requests.post(ML_API_URL, json=averages, timeout=10)
            resp.raise_for_status()
            result = resp.json()  # expects {"verdict": "good"/"bad", "confidence": 0.92}
        except Exception as e:
            result = {"verdict": "error", "confidence": 0.0, "error": str(e)}

        with self.lock:
            self.last_result = {
                **averages,
                **result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.status = "complete"
            self.notification = "Scan complete! Check the Results tab."

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
    """Cached so the MQTT connection + background thread is created exactly once
    for the whole app process, not re-created on every Streamlit rerun."""

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
    client.loop_start()  # runs in its own background thread
    return client


def publish_scan_command(client: mqtt.Client, sensor_key: str):
    client.publish(CMD_TOPICS[sensor_key], "1")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sensor Scan Dashboard", layout="wide")

state = get_state()
client = get_mqtt_client(state)

# Poll every second so MQTT updates (which happen on a background thread)
# get reflected in the UI without the user needing to click anything.
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
                publish_scan_command(client, key)
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
