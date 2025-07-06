import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import logging
import sys
from pathlib import Path


def plot_point_cloud_with_joints(points, joints, joints_true):
    x_range = [min(points[:, 0].min(), joints[:, 0].min(), joints_true[:, 0].min()), 
               max(points[:, 0].max(), joints[:, 0].max(), joints_true[:, 0].max())]
    y_range = [min(points[:, 1].min(), joints[:, 1].min(), joints_true[:, 1].min()), 
               max(points[:, 1].max(), joints[:, 1].max(), joints_true[:, 1].max())]
    z_range = [min(points[:, 2].min(), joints[:, 2].min(), joints_true[:, 2].min()), 
               max(points[:, 2].max(), joints[:, 2].max(), joints_true[:, 2].max())]

    range_max = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1, label='Point Cloud')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=30, label='Joints')
    ax.scatter(joints_true[:, 0], joints_true[:, 1], joints_true[:, 2], c='g', s=30, label='Target Joints', marker='^')
    ax.legend()
    ax.set_title('Point Cloud with Joints')
    ax.set_xlim([x_range[0], x_range[0] + range_max])
    ax.set_ylim([y_range[0], y_range[0] + range_max])
    ax.set_zlim([z_range[0], z_range[0] + range_max])
    plt.show()

def setup_directories(args):
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    exp_dir = Path('./models/classifier/log/part_seg')
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_dir.joinpath(args.log_dir or timestr)
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    return exp_dir

def setup_logger(exp_dir, model_name):
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(exp_dir / 'logs' / f'{model_name}.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger