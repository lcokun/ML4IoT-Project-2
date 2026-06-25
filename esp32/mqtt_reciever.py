import json
import paho.mqtt.client as mqtt
from datetime import datetime

# ================= CONFIG =================
BROKER_IP   = "localhost"
BROKER_PORT = 1883
DATA_FILE   = "latest_data.json"

TOPICS = [
    "health/ecg",
    "health/bpm",
    "health/temperature",
    "health/spo2",
]

# ================= STATE =================
latest = {
    "raw":         None,
    "bpm":         None,
    "temperature": None,
    "spo2":        None,
    "timestamp":   None,
}

def save_to_file():
    with open(DATA_FILE, "w") as f:
        json.dump(latest, f)

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

    latest["timestamp"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    if key == "ecg":
        latest["raw"] = data.get("raw")
    elif key == "bpm":
        latest["bpm"] = data.get("bpm")
    elif key == "temperature":
        latest["temperature"] = data.get("temperature")
    elif key == "spo2":
        latest["spo2"] = data.get("spo2")

    save_to_file()

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