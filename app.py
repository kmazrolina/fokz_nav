import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# Add ronin source to path
sys.path.append(os.path.join( "source"))

from ronin_resnet import get_model  # make sure get_model is imported correctly

# ---- Load model globally ----
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = "ronin_resnet/checkpoint_gsn_latest.pt"

checkpoint = torch.load(model_path, map_location=device)
network = get_model('resnet18')
network.load_state_dict(checkpoint['model_state_dict'])
network.eval().to(device)
print(f'Model {model_path} loaded to device {device}.')

# ---- Flask app ----
app = Flask(__name__)

def get_new_position(start_pos, imu_sample, dt=1.0):
    """
    Compute new position from start_pos and imu_sample using the RoNIN model
    """
    feat = torch.from_numpy(np.array(imu_sample, dtype=np.float32)).to(device)

    start_time = time.time()
    with torch.no_grad():
        pred = network(feat)  # (1,2) -> vx, vy
    inference_time = time.time() - start_time

    vx, vy = pred.cpu().numpy()[0]
    new_pos = np.array(start_pos) + np.array([vx, vy]) * dt



    return new_pos.tolist(), vx, vy, inference_time


EXPECTED_SAMPLES = 200
DT_NS = 5_000_000  # 5 ms in nanoseconds

@app.route('/predict_position', methods=['POST'])
def predict_position():
    data = request.json
    if 'start' not in data or 'imu_data' not in data:
        return jsonify({"error": "Request must contain 'start' and 'imu_data'"}), 400

    start_pos = np.array(data['start'], dtype=np.float32)
    imu_records = data['imu_data']

    # Extract timestamps, acc, gyro into arrays
    t_ns = np.array([rec['t_ns'] for rec in imu_records], dtype=np.int64)
    acc = np.array([rec['accc'] for rec in imu_records], dtype=np.float32)  # shape (N,3)
    gyro = np.array([rec['gyro'] for rec in imu_records], dtype=np.float32)  # shape (N,3)

    N = len(t_ns)

    if N != EXPECTED_SAMPLES:
        # Interpolate to exactly 200 samples spaced by 5ms
        t_target = np.arange(t_ns[0], t_ns[0] + EXPECTED_SAMPLES * DT_NS, DT_NS)
        acc_interp = np.zeros((EXPECTED_SAMPLES, 3), dtype=np.float32)
        gyro_interp = np.zeros((EXPECTED_SAMPLES, 3), dtype=np.float32)

        for i in range(3):
            f_acc = interp1d(t_ns, acc[:, i], kind='linear', fill_value='extrapolate')
            f_gyro = interp1d(t_ns, gyro[:, i], kind='linear', fill_value='extrapolate')
            acc_interp[:, i] = f_acc(t_target)
            gyro_interp[:, i] = f_gyro(t_target)

        acc = acc_interp
        gyro = gyro_interp
        t_ns = t_target

    # Stack into (1,6,200) as expected by model
    imu_sample = np.zeros((1, 6, EXPECTED_SAMPLES), dtype=np.float32)
    imu_sample[0, :3, :] = acc.T
    imu_sample[0, 3:, :] = gyro.T
    print(f"Prepared IMU sample shape: {imu_sample.shape}")

    

    try:
        start_time = time.time()
        new_pos = get_new_position(start_pos, imu_sample)
        inference_time = time.time() - start_time
        
        print(len(new_pos))
        
        return jsonify({
            "new_position": [float(x) for x in new_pos[0]],
            "inference_time_s": inference_time
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
