import sys
import os
import time
import torch
import numpy as np
from flask import Flask, request, jsonify
from scipy.interpolate import interp1d

# Add ronin source to path
sys.path.append(os.path.join("source"))

from ronin_resnet import get_model  # make sure get_model is imported correctly

# ---- Load model globally ----
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = "ronin_resnet/checkpoint_gsn_latest.pt"

checkpoint = torch.load(model_path, map_location=device)
network = get_model('resnet18')
network.load_state_dict(checkpoint['model_state_dict'])
network.eval().to(device)
print(f'Model {model_path} loaded to device {device}.')

EXPECTED_SAMPLES = 200
DT_NS = 5_000_000  # 5 ms in nanoseconds

# ---- Flask app ----
app = Flask(__name__)


def resample_imu(t_ns, acc, gyro, expected_samples=EXPECTED_SAMPLES, dt_ns=DT_NS):
    """
    Resample IMU data to exactly `expected_samples`.
    - If too few samples -> interpolate linearly
    - If too many samples -> average down to target length
    """
    N = len(t_ns)

    if N < expected_samples:
        # --- interpolate up to expected_samples ---
        t_target = np.arange(t_ns[0], t_ns[0] + expected_samples * dt_ns, dt_ns)
        acc_interp = np.zeros((expected_samples, 3), dtype=np.float32)
        gyro_interp = np.zeros((expected_samples, 3), dtype=np.float32)

        for i in range(3):
            f_acc = interp1d(t_ns, acc[:, i], kind='linear', fill_value='extrapolate')
            f_gyro = interp1d(t_ns, gyro[:, i], kind='linear', fill_value='extrapolate')
            acc_interp[:, i] = f_acc(t_target)
            gyro_interp[:, i] = f_gyro(t_target)

        return t_target, acc_interp, gyro_interp

    elif N > expected_samples:
        # --- downsample by averaging into bins ---
        # Find how many samples per bin (approx)
        step = N / expected_samples
        indices = (np.arange(expected_samples) * step).astype(int)

        acc_down = np.zeros((expected_samples, 3), dtype=np.float32)
        gyro_down = np.zeros((expected_samples, 3), dtype=np.float32)
        t_down = np.zeros(expected_samples, dtype=np.int64)

        for i in range(expected_samples):
            start = int(i * step)
            end = int((i + 1) * step)
            if end <= start:
                end = start + 1
            acc_down[i] = acc[start:end].mean(axis=0)
            gyro_down[i] = gyro[start:end].mean(axis=0)
            t_down[i] = t_ns[start:end].mean().astype(np.int64)

        return t_down, acc_down, gyro_down

    else:
        # Already the right length
        return t_ns, acc, gyro


def get_new_position(start_pos, imu_sample, dt=1.0):
    """
    Compute new position from start_pos and imu_sample using the RoNIN model
    """
    feat = torch.from_numpy(np.array(imu_sample, dtype=np.float32)).to(device)

    with torch.no_grad():
        pred = network(feat)  # (1,2) -> vx, vy

    vx, vy = pred.cpu().numpy()[0]
    new_pos = np.array(start_pos) + np.array([vx, vy]) * dt

    return new_pos.tolist()


@app.route('/predict_position', methods=['POST'])
def predict_position():
    data = request.json
    if 'start' not in data or 'imu_data' not in data:
        return jsonify({"error": "Request must contain 'start' and 'imu_data'"}), 400

    
    print(f"Received request {data["imu_data"][0].keys()} with {len(data['imu_data'])} IMU samples.")
    start_pos = np.array(data['start'], dtype=np.float32)
    imu_records = data['imu_data']

    valid_records = []
    for rec in imu_records:
        if 't_ns' in rec and 'acc' in rec and 'gyro' in rec:
            # ensure they have the correct length too
            if len(rec['acc']) == 3 and len(rec['gyro']) == 3:
                valid_records.append(rec)

    if len(valid_records) < 2:
        return jsonify({"error": "Not enough valid IMU records"}), 400

    # Now convert to numpy arrays
    t_ns = np.array([rec['t_ns'] for rec in valid_records], dtype=np.int64)
    acc = np.array([rec['acc'] for rec in valid_records], dtype=np.float32)   # (N, 3)
    gyro = np.array([rec['gyro'] for rec in valid_records], dtype=np.float32) # (N, 3)

    # Resample to exactly EXPECTED_SAMPLES
    t_ns, acc, gyro = resample_imu(t_ns, acc, gyro, EXPECTED_SAMPLES, DT_NS)

    # Stack into (1,6,200) as expected by model
    imu_sample = np.zeros((1, 6, EXPECTED_SAMPLES), dtype=np.float32)
    imu_sample[0, :3, :] = acc.T
    imu_sample[0, 3:, :] = gyro.T
    print(f"Prepared IMU sample shape: {imu_sample.shape}")

    try:
        start_time = time.time()
        new_pos = get_new_position(start_pos, imu_sample)
        inference_time = time.time() - start_time
        
        print(f"Predicted new position: {new_pos} (inference time: {inference_time:.6f} s)")

        return jsonify({
            "new_position": [float(x) for x in new_pos],
            "inference_time_s": float(inference_time)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/test_inference', methods=['GET'])
def test_inference():
    # generate mock data
    imu_sample = np.random.randn(1, 6, 200).astype(np.float32) * 0.1
    start_pos = np.random.rand(2) * 10

    print("==== TEST INFERENCE ====")
    print("Start position:", start_pos)
    print("Mock IMU sample shape:", imu_sample.shape)
    print("Sample values (first timestep of first channel):", imu_sample[0, 0, 0])

    # perform prediction
    new_pos, vx, vy, inference_time = get_new_position(start_pos, imu_sample)

    print("Predicted velocities: vx={:.6f}, vy={:.6f}".format(vx, vy))
    print("New position:", new_pos)
    print("Inference time: {:.6f} s".format(inference_time))
    print("========================")

    return jsonify({
        "start_position": start_pos.tolist(),       # already converted to list
        "new_position": [float(x) for x in new_pos],
        "vx": float(vx),
        "vy": float(vy),
        "inference_time_s": float(inference_time),
        "plot_path": "trajectory_plot.png"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
