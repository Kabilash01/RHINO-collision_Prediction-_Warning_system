import streamlit as st
import socket
import threading
import time 

st.set_page_config(page_title="RHINO-X Dashboard", layout="centered")
st.title("📡 RHINO-X Live Sensor Monitor")
status_placeholder = st.empty()
temp_display = st.metric(label="🌡️ Temperature (°C)", value="--")
hum_display = st.metric(label="💧 Humidity (%)", value="--")
alert_display = st.empty()

sensor_data = {"temp": "--", "hum": "--", "status": "Waiting for ESP32...", "alert": ""}

def socket_server():
    global sensor_data
    HOST = '0.0.0.0'
    PORT = 8888
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    conn, addr = server.accept()

    sensor_data["status"] = f"✅ Connected to ESP32 ({addr[0]})"

    while True:
        try:
            data = conn.recv(1024).decode().strip()
            if not data:
                continue
            temp, hum = map(float, data.split(","))
            sensor_data["temp"] = round(temp, 1)
            sensor_data["hum"] = round(hum, 1)

            if temp > 35.0 or hum > 80.0:
                conn.sendall(b"CRASH\n")
                sensor_data["alert"] = "🚨 CRASH ALERT TRIGGERED!"
            else:
                conn.sendall(b"SAFE\n")
                sensor_data["alert"] = "✅ All Clear"

        except Exception as e:
            sensor_data["status"] = f"❌ Disconnected: {e}"
            break

threading.Thread(target=socket_server, daemon=True).start()

# Live UI update loo
while True:
    status_placeholder.info(sensor_data["status"])
    temp_display.metric("🌡️ Temperature (°C)", sensor_data["temp"])
    hum_display.metric("💧 Humidity (%)", sensor_data["hum"])
    alert_display.warning(sensor_data["alert"])
    time.sleep(1)
