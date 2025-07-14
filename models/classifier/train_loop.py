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
from pathlib import Path
from tqdm import tqdm
from dataset.dataset_pointnet import TruebonesDataset, pc_normalize, load_datasets
import torch.nn as nn
from utils.pointnet_utils import setup_directories, setup_logger, \
    initialize_model, initialize_optimizer, adjust_learning_rate_and_momentum
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def train_one_epoch(model, optimizer, train_loader, criterion, epoch):
    model.train()
    mean_correct = []
    losses_l1 = []
    losses_skin = []

    for points, label, mask, target, weights_gt, tpl_edge_index, N, _ in tqdm(train_loader, smoothing=0.9):
        optimizer.zero_grad()

        # Augmentations
        points, weights_gt = provider.shuffle_points(points, weights_gt)

        # Move on gpu
        points, label, mask, target, weights_gt = points.cuda().float(), label.cuda().long(), mask.cuda(), target.cuda(), weights_gt.cuda()
        points = points.transpose(2, 1)

        # Forward pass and loss calculation
        joints_pred, skin_weights, cls_logits = model(points, tpl_edge_index, mask)
        joints_pred *= mask.unsqueeze(-1).expand_as(joints_pred)
        target *= mask.unsqueeze(-1).expand_as(target)
        loss, all_losses = criterion(cls_logits, label, joints_pred, target, mask, skin_weights, weights_gt)
        l1_loss = all_losses['joint_loss']
        skin_loss = all_losses['skin_loss']

        correct = torch.sum(torch.max(cls_logits, 1)[1] == label).item()
        mean_correct.append(correct / float(points.size()[0]))
        losses_l1.append(l1_loss)
        losses_skin.append(skin_loss)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    return np.mean(mean_correct), np.mean(losses_l1), np.mean(losses_skin)

def test_one_epoch(model, test_loader, criterion, epoch):
    model.eval()
    losses = []
    mean_correct = []
    losses_l1 = []
    losses_skin = []

    for points, label, mask, target, weights_gt, tpl_edge_index, N, _ in tqdm(test_loader, smoothing=0.9):

        # Augmentations
        points, weights_gt = provider.shuffle_points(points, weights_gt)

        # Move on gpu
        points, label, mask, target, weights_gt = points.cuda().float(), label.cuda().long(), mask.cuda(), target.cuda(), weights_gt.cuda()
        points = points.transpose(2, 1)

        # Forward pass and loss calculation
        joints_pred, skin_weights, cls_logits = model(points, tpl_edge_index, mask)
        joints_pred *= mask.unsqueeze(-1).expand_as(joints_pred)
        target *= mask.unsqueeze(-1).expand_as(target)
        loss, all_losses = criterion(cls_logits, label, joints_pred, target, mask, skin_weights, weights_gt)
        l1_loss = all_losses['joint_loss']
        skin_loss = all_losses['skin_loss']

        correct = torch.sum(torch.max(cls_logits, 1)[1] == label).item()
        mean_correct.append(correct / float(points.size()[0]))
        losses_l1.append(l1_loss)
        losses_skin.append(skin_loss)
        losses.append(loss.item())

    return np.mean(mean_correct), np.mean(losses_l1), np.mean(losses_skin), np.mean(losses)

def monte_carlo_evaluation(model, test_loader, criterion, epoch, num_repetitions_eval):
    model.eval()
    losses = [0]
    mean_correct = []
    losses_l1 = []

    for _, _, _, _, _, _, _, indexes in tqdm(test_loader, smoothing=0.9):
        all_classes = []
        all_joints_pred = []

        for _ in range(num_repetitions_eval):

            batch = []
            for i in range(len(indexes)):
                element = test_loader.dataset[int(indexes[i])]
                batch.append(element)
            batch = torch.utils.data.default_collate(batch)

            points, label, mask, target, weights_gt, tpl_edge_index, N, _ = batch

            # Augmentations
            points, weights_gt = provider.shuffle_points(points, weights_gt)

            # Move on gpu
            points, label, mask, target, weights_gt = points.cuda().float(), label.cuda().long(), mask.cuda(), target.cuda(), weights_gt.cuda()
            points = points.transpose(2, 1)

            # Forward pass and loss calculation
            joints_pred, skin_weights, cls_logits = model(points, tpl_edge_index, mask)
            joints_pred *= mask.unsqueeze(-1).expand_as(joints_pred)
            target *= mask.unsqueeze(-1).expand_as(target)
            
            loss, all_losses = criterion(cls_logits, label, joints_pred, target, mask, skin_weights, weights_gt)
            l1_loss = all_losses['joint_loss']

            all_classes.append(torch.max(cls_logits, 1)[1].cpu().numpy())
            all_joints_pred.append(joints_pred)

        all_classes = np.array(all_classes)
        all_joints_pred = torch.stack(all_joints_pred, dim=0)

        # Calculate the most common class
        mode_classes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_classes)
        correct = np.sum(mode_classes == label.cpu().numpy())
        mean_correct.append(correct / float(points.size()[0]))

        # take the median
        avg_joints = all_joints_pred.median(dim=0).values
        avg_joints *= mask.unsqueeze(-1).expand_as(avg_joints)
        target *= mask.unsqueeze(-1).expand_as(target)

        loss, all_losses = criterion(cls_logits, label, avg_joints, target, mask, skin_weights, weights_gt)
        l1_loss = all_losses['joint_loss']

        losses_l1.append(l1_loss)

    return np.mean(mean_correct), np.mean(losses_l1)


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
    best_loss = torch.inf
    for epoch in range(start_epoch, args.epoch):
        lr, momentum = adjust_learning_rate_and_momentum(optimizer, model, epoch, args)
        logger.info(f'Epoch {epoch + 1}: learning rate={lr:.6f}, BN momentum={momentum:.6f}')

        acc_train, lossl1_train, loss_skin_train = train_one_epoch(model, optimizer, train_loader, criterion, epoch)
        acc_test, lossl1_test, loss_skin_test, all_loss = test_one_epoch(model, test_loader, criterion, epoch)

        if all_loss < best_loss:
            best_loss = all_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': all_loss,
            }, exp_dir / 'checkpoints/best_model.pth')
            logger.info(f'Saving best model at epoch {epoch + 1} with loss {all_loss:.6f}')

        logger.info(f'Train L1 Loss: {lossl1_train:.6f}, Accuracy: {acc_train:.6f}, Skin Loss: {loss_skin_train:.6f}')
        logger.info(f'Test L1 Loss: {lossl1_test:.6f}, Accuracy: {acc_test:.6f}, Skin Loss: {loss_skin_test:.6f}')

        if epoch % args.eval_freq == 0 or epoch == args.epoch - 1:
            acc_test, lossl1_test = monte_carlo_evaluation(model, test_loader, criterion, epoch, args.num_repetitions_eval)
            print(f'Monte Carlo Evaluation - L1 Loss: {lossl1_test:.6f}, Accuracy: {acc_test:.6f}')
