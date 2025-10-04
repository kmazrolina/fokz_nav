import requests
import numpy as np
import time

# URL of your Flask server (update if running on a different host/port)
URL = "http://127.0.0.1:5000/predict_position"

# Generate mock start position
start_pos = [np.random.rand() * 10, np.random.rand() * 10]

# Generate mock IMU data (timestamps every 5ms, 200 samples)
N_SAMPLES = 200
DT_NS = 5_000_000  # 5 ms in nanoseconds

t0_ns = int(time.time() * 1e9)
timestamps = t0_ns + np.arange(N_SAMPLES) * DT_NS

imu_data = []
for t in timestamps:
    acc_sample = (np.random.randn(3) * 0.1).tolist()
    gyro_sample = (np.random.randn(3) * 0.1).tolist()
    imu_data.append({
        "t_ns": int(t),
        "accc": acc_sample,
        "gyro": gyro_sample
    })

# Compose the request JSON
payload = {
    "start": start_pos,
    "imu_data": imu_data
}

# Optional: include ngrok header if needed
headers = {
    "ngrok-skip-browser-warning": "true"
}

# Send POST request
response = requests.post(URL, json=payload, headers=headers)

# Print results
if response.ok:
    data = response.json()
    print("Start position:", start_pos)
    print("Predicted new position:", data["new_position"])
    print("Inference time (s):", data["inference_time_s"])
else:
    print("Error:", response.status_code, response.text)
