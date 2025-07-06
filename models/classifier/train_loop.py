import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np
import models.classifier.provider as provider
from models.classifier.models.pointnet2 import PointNetPartSeg, JointPredictionLoss
from pathlib import Path
from tqdm import tqdm
from dataset.dataset_pointnet import TruebonesDataset, pc_normalize, load_datasets
import torch.nn as nn
from utils.pointnet_utils import setup_directories, setup_logger, plot_point_cloud_with_joints

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def initialize_model(args, num_classes, seg_classes, exp_dir, logger):
    model = PointNetPartSeg(num_classes, seg_classes).cuda()
    criterion = JointPredictionLoss().cuda()

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

def train_one_epoch(model, optimizer, train_loader, criterion):
    model.train()
    mean_correct = []
    losses = []
    for points, label, mask, target in tqdm(train_loader, smoothing=0.9):
        optimizer.zero_grad()
        points = provider.shuffle_points(points)
        points_joint = provider.shift_point_cloud(provider.random_scale_point_cloud(np.concatenate((points, target), axis=1)))
        points = torch.Tensor(points_joint[:, :points.shape[1], :])
        target = torch.Tensor(points_joint[:, points.shape[1]:, :])
        points, label, mask, target = points.cuda().float(), label.cuda().long(), mask.cuda(), target.cuda()
        points = points.transpose(2, 1)

        joints_pred, cls_logits = model(points)
        joints_pred *= mask.unsqueeze(-1).expand_as(joints_pred)
        target *= mask.unsqueeze(-1).expand_as(target)

        loss = criterion(cls_logits, label, joints_pred, target, mask)
        loss.backward()
        optimizer.step()

        correct = torch.sum(torch.max(cls_logits, 1)[1] == label).item()
        mean_correct.append(correct / float(points.size()[0]))

        l1_loss = torch.sum(torch.abs(joints_pred - target)) / (mask.sum() * 3)
        losses.append(l1_loss.item())
    return np.mean(mean_correct), np.mean(losses)

def evaluate_model(model, test_loader, criterion, epoch):
    model.eval()
    mean_correct = []
    losses = []
    with torch.no_grad():
        for points, label, mask, target in tqdm(test_loader, smoothing=0.9):
            points = provider.shuffle_points(points)
            points_joint = provider.shift_point_cloud(provider.random_scale_point_cloud(np.concatenate((points, target), axis=1)))
            points = torch.Tensor(points_joint[:, :points.shape[1], :])
            target = torch.Tensor(points_joint[:, points.shape[1]:, :])
            points, label, mask, target = points.cuda().float(), label.cuda().long(), mask.cuda(), target.cuda()
            points = points.transpose(2, 1)
            joints_pred, cls_logits = model(points)
            joints_pred *= mask.unsqueeze(-1).expand_as(joints_pred)
            target *= mask.unsqueeze(-1).expand_as(target)

            correct = torch.sum(torch.max(cls_logits, 1)[1] == label).item()
            mean_correct.append(correct / float(points.size()[0]))

            l1_loss = torch.sum(torch.abs(joints_pred - target)) / (mask.sum() * 3)
            losses.append(l1_loss.item())

            if epoch > 197:
                pc = points[0].transpose(0, 1).cpu().numpy()
                joints = joints_pred[0][mask[0].bool()].cpu().numpy()
                joints_true = target[0][mask[0].bool()].cpu().numpy()
                plot_point_cloud_with_joints(pc, joints, joints_true)

    return np.mean(mean_correct), np.mean(losses)

def train_loop(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    exp_dir = setup_directories(args)
    logger = setup_logger(exp_dir, "pointnet2")
    logger.info('PARAMETER ...')
    logger.info(args)

    train_dataset, test_dataset, train_loader, test_loader = load_datasets(args)
    logger.info(f"The number of training data is: {len(train_dataset)}")
    logger.info(f"The number of test data is: {len(test_dataset)}")

    model, criterion, start_epoch = initialize_model(args, train_dataset.get_num_classes(), train_dataset.get_num_part(), exp_dir, logger)
    optimizer = initialize_optimizer(args, model)

    for epoch in range(start_epoch, args.epoch):
        lr, momentum = adjust_learning_rate_and_momentum(optimizer, model, epoch, args)
        logger.info(f'Epoch {epoch + 1}: learning rate={lr:.6f}, BN momentum={momentum:.6f}')

        acc_train, loss_train = train_one_epoch(model, optimizer, train_loader, criterion)
        acc_test, loss_test = evaluate_model(model, test_loader, criterion, epoch)

        logger.info(f'Train L1 Loss: {loss_train:.6f}, Accuracy: {acc_train:.6f}')
        logger.info(f'Test L1 Loss: {loss_test:.6f}, Accuracy: {acc_test:.6f}')