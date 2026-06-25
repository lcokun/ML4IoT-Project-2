#include <Arduino.h>
#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"

#include <OneWire.h>
#include <DallasTemperature.h>

#include <WiFi.h>
#include <PubSubClient.h>

// ================= WIFI & MQTT =================
const char* ssid       = "BuayaHaute";
const char* password   = "A24AI9471";
const char* mqttServer = "192.168.100.218";
const int   mqttPort   = 1883;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// ================= PINS =================
#define ONE_WIRE_BUS 5
const int ECG_PIN  = 34;
const int LO_PLUS  = 25;
const int LO_MINUS = 26;

// ================= TEMPERATURE =================
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// ================= SPO2 =================
MAX30105 particleSensor;

#define BUFFER_SIZE 100

uint32_t irBuffer[BUFFER_SIZE];
uint32_t redBuffer[BUFFER_SIZE];

int32_t spo2;
int8_t  validSPO2;
int32_t heartRate;
int8_t  validHeartRate;
int     spo2BufferIndex = 0;

// ================= ECG BPM =================
unsigned long lastBeatTime = 0;
int  bpm          = 0;
int  ecgThreshold = 2000;
bool beatDetected = false;

// ================= TIMING =================
unsigned long lastTempTime = 0;
unsigned long lastEcgTime  = 0;

// ================= WIFI =================
void connectWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected. IP: ");
  Serial.println(WiFi.localIP());
}

// ================= MQTT =================
void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Connecting to MQTT...");
    if (mqttClient.connect("ESP32Client")) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.println(mqttClient.state());
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);

  // ================= ECG =================
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  // ================= TEMPERATURE =================
  sensors.begin();

  // ================= WIFI & MQTT =================
  connectWiFi();
  mqttClient.setServer(mqttServer, mqttPort);
  connectMQTT();

  // ================= SPO2 =================
  Wire.begin(21, 22);

  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 NOT FOUND");
    while (1);
  }

  particleSensor.setup(
    60,    // LED brightness
    4,     // sample average
    2,     // red + IR
    100,   // sample rate
    411,   // pulse width
    4096   // ADC range
  );

  Serial.println("MAX30102 ready. Place finger.");
}

void loop() {

  // keep MQTT alive
  if (!mqttClient.connected()) connectMQTT();
  mqttClient.loop();

  // =====================================================
  // ECG
  // =====================================================
  if (millis() - lastEcgTime >= 5) {
    lastEcgTime = millis();

    int ecgValue;
    if (digitalRead(LO_PLUS) || digitalRead(LO_MINUS)) {
      ecgValue = 0;
    } else {
      ecgValue = analogRead(ECG_PIN);
    }

    Serial.print(">val:");
    Serial.println(ecgValue);

    char ecgJson[64];
    snprintf(ecgJson, sizeof(ecgJson), "{\"raw\":%d}", ecgValue);
    mqttClient.publish("health/ecg", ecgJson);

    // BPM detection
    if (ecgValue > ecgThreshold && !beatDetected) {
      beatDetected = true;

      unsigned long now      = millis();
      unsigned long interval = now - lastBeatTime;

      if (interval > 300 && interval < 2000) {
        bpm = 60000 / interval;
        Serial.print("BPM=");
        Serial.println(bpm);

        char bpmJson[64];
        snprintf(bpmJson, sizeof(bpmJson), "{\"bpm\":%d}", bpm);
        mqttClient.publish("health/bpm", bpmJson);
      }

      lastBeatTime = now;
    }

    if (ecgValue < ecgThreshold - 100) {
      beatDetected = false;
    }
  }

  // =====================================================
  // TEMPERATURE
  // =====================================================
  if (millis() - lastTempTime >= 1000) {
    lastTempTime = millis();

    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);

    Serial.print("Temperature: ");
    Serial.println(tempC);

    char tempJson[64];
    snprintf(tempJson, sizeof(tempJson), "{\"temperature\":%.2f}", tempC);
    mqttClient.publish("health/temperature", tempJson);
  }

  // =====================================================
  // SPO2
  // =====================================================
  if (millis() - lastEcgTime < 4) {
    particleSensor.check();

    if (particleSensor.available()) {
      redBuffer[spo2BufferIndex] = particleSensor.getRed();
      irBuffer[spo2BufferIndex]  = particleSensor.getIR();
      particleSensor.nextSample();
      spo2BufferIndex++;

      if (spo2BufferIndex >= BUFFER_SIZE) {
        spo2BufferIndex = 0;

        maxim_heart_rate_and_oxygen_saturation(
          irBuffer, BUFFER_SIZE,
          redBuffer,
          &spo2, &validSPO2,
          &heartRate, &validHeartRate
        );

        Serial.print("SpO2=");
        if (validSPO2) {
          Serial.println(spo2);
          char spo2Json[64];
          snprintf(spo2Json, sizeof(spo2Json), "{\"spo2\":%d}", (int)spo2);
          mqttClient.publish("health/spo2", spo2Json);
        } else {
          Serial.println("Invalid");
        }
      }
    }
  }
}