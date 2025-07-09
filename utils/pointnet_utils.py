import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import logging
import sys
from pathlib import Path
from models.classifier.models.pointnet2 import PointNetPartSeg, SkinningLoss
import torch.nn as nn
import torch.nn.functional as F

def initialize_model(args, num_classes, seg_classes, exp_dir, logger):
    model = PointNetPartSeg(num_classes, seg_classes).cuda()
    criterion = SkinningLoss().cuda()

    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(exp_dir / 'checkpoints/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        model.apply(weights_init)
        start_epoch = 0

    return model, criterion, start_epoch

def initialize_optimizer(args, model):
    if args.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

def adjust_learning_rate_and_momentum(optimizer, model, epoch, args):
    lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), 1e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    momentum = max(0.1 * (0.5 ** (epoch // args.step_size)), 0.01)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.momentum = momentum
    return lr, momentum

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