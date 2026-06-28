import json
import time
from collections import deque

import paho.mqtt.client as mqtt
from datetime import datetime

# ================= CONFIG =================
BROKER_IP   = "localhost"
BROKER_PORT = 1883
DATA_FILE   = "latest_data.json"

LOG_FILE     = "data_log.json"  # accumulating history, separate from the "latest snapshot only" DATA_FILE
LOG_INTERVAL_S = 60              # how often a snapshot gets appended to LOG_FILE

ECG_WINDOW_SIZE = 200  # rolling buffer of raw ECG samples for the waveform plot

TOPICS = [
    "health/ecg",
    "health/bpm",
    "health/temperature",
    "health/spo2",
]

# ================= STATE =================
# "ecg_window" buffers raw samples over time so the dashboard has something
# to plot -- a single "raw" value would just overwrite itself on every
# message and never show a waveform.
ecg_window = deque(maxlen=ECG_WINDOW_SIZE)

latest = {
    "raw":         None,   # most recent single ECG sample (debug/back-compat)
    "ecg_window":  [],     # rolling buffer of recent ECG samples for plotting
    "bpm":         None,
    "temperature": None,
    "spo2":        None,
    "timestamp":   None,   # human-readable, for console/debugging
    "received_at": None,   # epoch seconds, for staleness checks in Streamlit
}

last_log_time = time.time()  # gate for append_log_entry(); first entry written LOG_INTERVAL_S after startup

def save_to_file():
    with open(DATA_FILE, "w") as f:
        json.dump(latest, f)

def append_log_entry():
    """Append a snapshot of the current reading to LOG_FILE, at most once
    every LOG_INTERVAL_S seconds. This is what gives you an actual record
    over time -- DATA_FILE only ever holds the single most recent reading
    and gets overwritten on every message, so nothing there persists.
    """
    global last_log_time
    now = time.time()
    if now - last_log_time < LOG_INTERVAL_S:
        return
    last_log_time = now

    try:
        with open(LOG_FILE, "r") as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = []

    log.append(dict(latest))  # shallow copy -- latest's values aren't mutated in place

    with open(LOG_FILE, "w") as f:
        json.dump(log, f)

    print(f"Logged snapshot to {LOG_FILE} ({len(log)} entries so far)")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to broker at {BROKER_IP}:{BROKER_PORT}")
        for topic in TOPICS:
            client.subscribe(topic)
            print(f"  Subscribed to {topic}")
        print()
    else:
        print(f"Connection failed. Code: {rc}")

def on_message(client, userdata, msg):
    topic = msg.topic
    key   = topic.split("/")[-1]

    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        return

    latest["timestamp"]   = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    latest["received_at"] = time.time()

    if key == "ecg":
        raw_value = data.get("raw")
        latest["raw"] = raw_value
        if raw_value is not None:
            ecg_window.append(raw_value)
            latest["ecg_window"] = list(ecg_window)
    elif key == "bpm":
        latest["bpm"] = data.get("bpm")
    elif key == "temperature":
        latest["temperature"] = data.get("temperature")
    elif key == "spo2":
        latest["spo2"] = data.get("spo2")

    save_to_file()
    append_log_entry()

# ================= MAIN =================
def main():
    client = mqtt.Client(client_id="mqtt_receiver")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER_IP, BROKER_PORT, keepalive=60)
    except Exception as e:
        print(f"Could not connect: {e}")
        return

    print("Receiver running...\n")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        client.disconnect()

if __name__ == "__main__":
    main()