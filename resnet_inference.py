import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join("ronin", "source"))  # uncomment and set path if necessary

from ronin_resnet import get_model  # your get_model() from the big script

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    network = get_model('resnet18')
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print(f'Model {model_path} loaded to device {device}.')
    return network

def get_new_position(start_pos, imu_sample, device, network, dt=1.0):
    """
    Given a start position and an IMU sample, predict the new position after dt seconds.
    imu_sample: numpy array of shape (1, 6, 200) - single IMU sample, accelerometer + gyroscope data of 200 timesteps (each 5ms = 200Hz sampling)
    start_pos: numpy array of shape (2,) - starting (x,y) position
    dt: time duration in seconds
    """
    # load imu sample to tensor (feature shape (1,6,200))
    feat = torch.from_numpy(imu_sample).to(device)
    
    with torch.no_grad():
        pred = network(feat)  # output shape (1,2) -> (vx, vy)
    vx, vy = pred.cpu().numpy()[0]
    print(f"Predicted velocity: vx={vx:.4f}, vy={vy:.4f}")

    # compute new position
    new_pos = start_pos + np.array([vx, vy]) * dt
    print(f"Computed new position: {new_pos}")
    
    return new_pos


def main():
    # pick device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load pretrained model
    model_path = "ronin/ronin_resnet/checkpoint_gsn_latest.pt"
    network = load_model(model_path, device)

    # ---- make minimal dummy dataset ----
    imu_sample = np.random.randn(1, 6, 200).astype(np.float32) * 0.1
    start_pos = np.random.rand(2) * 10  # random (x,y)

    
    print(f"Start position: {start_pos}")
    
    new_pos = get_new_position(start_pos, imu_sample, device, network, dt=1.0)
    print(f"New position after 1s: {new_pos}")

    # ---- plot and save ----
    plt.figure()
    plt.plot([start_pos[0], new_pos[0]], [start_pos[1], new_pos[1]], marker="o")
    plt.annotate("start", start_pos)
    plt.annotate("end", new_pos)
    plt.axis("equal")
    plt.title("Minimal RoNIN trajectory (1 second)")

    # save to disk
    output_path = "trajectory_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    # optionally show
    # plt.show()
    plt.close()

if __name__ == '__main__':
    main()
