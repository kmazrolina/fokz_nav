import requests
import numpy as np
import time
import random

# URL of your Flask server (update if needed)
URL = "https://1573cdc86738.ngrok-free.app/predict_position"

# Constants
N_SAMPLES = 200
DT_NS = 5_000_000  # 5 ms in nanoseconds

# Optional: ngrok header
HEADERS = {
    "ngrok-skip-browser-warning": "true"
}


def generate_imu_data(n_samples, dt_ns, drop_random=False, missing_keys=False):
    """
    Generate synthetic IMU data.
    - n_samples: number of samples to generate
    - dt_ns: time spacing in nanoseconds
    - drop_random: if True, randomly drop ~10% of samples
    - missing_keys: if True, randomly drop 'acc' or 'gyro' key from ~10% of samples
    """
    t0_ns = int(time.time() * 1e9)
    timestamps = t0_ns + np.arange(n_samples) * dt_ns

    imu_data = []
    for t in timestamps:
        rec = {
            "t_ns": int(t),
            "acc": (np.random.randn(3) * 0.1).tolist(),
            "gyro": (np.random.randn(3) * 0.1).tolist()
        }

        imu_data.append(rec)

    if drop_random:
        keep_count = int(len(imu_data) * 0.9)  # keep ~90%
        imu_data = random.sample(imu_data, keep_count)
        imu_data = sorted(imu_data, key=lambda x: x["t_ns"])  # maintain order

    if missing_keys:
        for rec in imu_data:
            if random.random() < 0.1:  # ~10% chance to remove a key
                if random.choice([True, False]):
                    rec.pop("acc", None)
                else:
                    rec.pop("gyro", None)

    return imu_data


def send_request(start_pos, imu_data, label):
    """Send request and print results."""
    payload = {
        "start": start_pos,
        "imu_data": imu_data
    }

    print(f"\n=== Scenario: {label} ===")
    response = requests.post(URL, json=payload, headers=HEADERS)

    if response.ok:
        data = response.json()
        print("Start position:", start_pos)
        print("Samples sent:", len(imu_data))
        print("Predicted new position:", data["new_position"])
        print("Inference time (s):", data["inference_time_s"])
    else:
        print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    # Random start position
    start_pos = [float(np.random.rand() * 10), float(np.random.rand() * 10)]

    # 1. Ideal (200 samples, 5ms apart)
    imu_data_ideal = generate_imu_data(N_SAMPLES, DT_NS)
    send_request(start_pos, imu_data_ideal, "Ideal (200 @ 5ms)")

    # 2. Missing datapoints (<200)
    imu_data_missing = generate_imu_data(N_SAMPLES, DT_NS, drop_random=True)
    send_request(start_pos, imu_data_missing, "Missing datapoints (<200)")

    # 3. Oversampled (>200 samples, 2.5ms spacing)
    imu_data_oversampled = generate_imu_data(400, DT_NS // 2)
    send_request(start_pos, imu_data_oversampled, "Oversampled (>200, 2.5ms)")

    # 4. Occasionally missing 'acc' or 'gyro' key
    imu_data_incomplete = generate_imu_data(N_SAMPLES, DT_NS, missing_keys=True)
    send_request(start_pos, imu_data_incomplete, "Incomplete records (missing 'acc' or 'gyro')")
