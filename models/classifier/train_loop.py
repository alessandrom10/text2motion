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

    for points, label, mask, target, weights_gt, tpl_edge_index, N in tqdm(train_loader, smoothing=0.9):
        optimizer.zero_grad()

        # Augmentations
        points, weights_gt = provider.shuffle_points(points, weights_gt)
        points_joint = provider.shift_point_cloud(provider.random_scale_point_cloud(np.concatenate((points, target), axis=1)))
        points = torch.Tensor(points_joint[:, :points.shape[1], :])
        target = torch.Tensor(points_joint[:, points.shape[1]:, :])

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

def evaluate_model(model, test_loader, criterion, epoch, num_repetitions_eval):
    model.eval()

    all_cls_logits = defaultdict(list)
    all_joints_preds = defaultdict(list)
    all_skin_losses = defaultdict(list)
    
    all_ground_truths_target = defaultdict(list)
    all_ground_truths_mask = defaultdict(list)
    all_ground_truths_label = defaultdict(list)

    with torch.no_grad():
        for run_idx in range(num_repetitions_eval):
            print(f"Evaluation run {run_idx + 1}/{num_repetitions_eval}")
            for batch_idx, (points, label, mask, target, weights_gt, tpl_edge_index, N) in enumerate(tqdm(test_loader)):
                
                # Augmentations
                points, weights_gt = provider.shuffle_points(points, weights_gt)
                points_joint = provider.shift_point_cloud(provider.random_scale_point_cloud(np.concatenate((points, target), axis=1)))
                points = torch.Tensor(points_joint[:, :points.shape[1], :])
                target = torch.Tensor(points_joint[:, points.shape[1]:, :])
                
                # Move on gpu
                points, label, mask, target, weights_gt = points.cuda().float(), label.cuda().long(), mask.cuda(), target.cuda(), weights_gt.cuda()
                points = points.transpose(2, 1)

                # Forward pass and loss calculation
                joints_pred, skin_weights, cls_logits = model(points, tpl_edge_index, mask)
                _, all_losses = criterion(cls_logits, label, joints_pred, target, mask, skin_weights, weights_gt)
                
                batch_size = points.size(0)
                for i in range(batch_size):
                    sample_id = batch_idx * batch_size + i
                    
                    all_cls_logits[sample_id].append(cls_logits[i])
                    all_joints_preds[sample_id].append(joints_pred[i])
                    all_skin_losses[sample_id].append(all_losses['skin_loss'])
                    
                    all_ground_truths_target[sample_id].append(target[i])
                    all_ground_truths_mask[sample_id].append(mask[i])
                    all_ground_truths_label[sample_id].append(label[i])

    total_correct = 0
    final_joint_losses = []
    num_samples = len(all_ground_truths_target)

    for i in range(num_samples):
        sample_logits = torch.stack(all_cls_logits[i], dim=0)
        sample_preds = torch.max(sample_logits, 1)[1]
        majority_vote, _ = torch.mode(sample_preds, dim=0)
        
        if majority_vote == all_ground_truths_label[i][0]:
            total_correct += 1
            
        sample_joints = torch.stack(all_joints_preds[i], dim=0)
        avg_joints_pred = sample_joints.mean(dim=0)
        
        losses_for_sample = []
        for k in range(num_repetitions_eval):
            gt_target_k = all_ground_truths_target[i][k]
            gt_mask_k = all_ground_truths_mask[i][k]
            
            avg_joints_pred_masked = avg_joints_pred * gt_mask_k.unsqueeze(-1).expand_as(avg_joints_pred)
            gt_target_k_masked = gt_target_k * gt_mask_k.unsqueeze(-1).expand_as(gt_target_k)
            
            joint_loss_k = torch.nn.functional.l1_loss(avg_joints_pred_masked, gt_target_k_masked)
            losses_for_sample.append(joint_loss_k.item())
        
        final_joint_losses.append(np.mean(losses_for_sample))

    flat_skin_losses = [loss for losses in all_skin_losses.values() for loss in losses]

    final_accuracy = total_correct / num_samples
    final_loss_l1 = np.mean(final_joint_losses)
    final_loss_skin = np.mean(flat_skin_losses)

    final_total_loss = final_loss_l1 + final_loss_skin
    
    return final_accuracy, final_loss_l1, final_loss_skin, final_total_loss



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
        logger.info(f'Train L1 Loss: {lossl1_train:.6f}, Accuracy: {acc_train:.6f}, Skin Loss: {loss_skin_train:.6f}')

        if epoch % args.eval_freq == 0 or epoch == args.epoch - 1:
            acc_test, lossl1_test, loss_skin_test, all_loss = evaluate_model(model, test_loader, criterion, epoch, args.num_repetitions_eval)

            if all_loss < best_loss:
                best_loss = all_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': all_loss,
                }, exp_dir / 'checkpoints/best_model.pth')
                logger.info(f'Saving best model at epoch {epoch + 1} with loss {all_loss:.6f}')

            logger.info(f'Test L1 Loss: {lossl1_test:.6f}, Accuracy: {acc_test:.6f}, Skin Loss: {loss_skin_test:.6f}')