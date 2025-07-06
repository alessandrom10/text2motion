import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np
import models.pointnet.provider as provider
from pathlib import Path
from tqdm import tqdm
from dataset.dataset_pointnet import TruebonesDataset, pc_normalize
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def plot_point_cloud_with_joints(points, joints, joints_true):

    # Get the ranges of the data
    x_range = [min(points[:, 0].min(), joints[:, 0].min(), joints_true[:, 0].min()), 
            max(points[:, 0].max(), joints[:, 0].max(), joints_true[:, 0].max())]
    y_range = [min(points[:, 1].min(), joints[:, 1].min(), joints_true[:, 1].min()), 
            max(points[:, 1].max(), joints[:, 1].max(), joints_true[:, 1].max())]
    z_range = [min(points[:, 2].min(), joints[:, 2].min(), joints_true[:, 2].min()), 
            max(points[:, 2].max(), joints[:, 2].max(), joints_true[:, 2].max())]

    # Set all axis limits to the same range
    range_max = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])


    # Plot
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

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch Size during training')
    parser.add_argument('--epoch', default=200, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)


    TRAIN_DATASET = TruebonesDataset(npoints=args.npoint, split='train', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = TruebonesDataset(npoints=args.npoint, split='test', normal_channel=args.normal)
    TEST_DATASET.set_seg_classes(TRAIN_DATASET.seg_classes)
    TEST_DATASET.set_part_classes(TRAIN_DATASET.part_classes)
    TEST_DATASET.set_dict_groups(TRAIN_DATASET.dict_groups)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = TRAIN_DATASET.get_num_classes()
    print(num_classes)
    seg_classes = TRAIN_DATASET.get_num_part()
    part_classes = TRAIN_DATASET.part_classes

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy("models/pointnet/models/" +args.model+'.py', str(exp_dir))
    shutil.copy("models/pointnet/models/" +args.model+'_utils.py', str(exp_dir))

    classifier = MODEL.PointNetPartSeg(num_classes, seg_classes).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0


    for epoch in range(start_epoch, args.epoch):
        mean_correct_train = []
        mean_correct_test = []
        losses_train = []
        losses_test = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, mask, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # Augmentation
            points = provider.shuffle_points(points)
            points_joint = np.concatenate((points, target), axis=1)
            #points_joint = provider.rotate_point_cloud(points_joint)
            points_joint = provider.random_scale_point_cloud(points_joint)
            points_joint = provider.shift_point_cloud(points_joint)

            points = torch.Tensor(points_joint[:, :points.shape[1], :]) 
            target = torch.Tensor(points_joint[:, points.shape[1]:, :])

            label = torch.Tensor(label)
            points, label, mask, target = points.cuda().float(), label.cuda().long(), mask.cuda(), target.cuda()
            points = points.transpose(2, 1)

            joints_pred, cls_logits, trans_feat, weights = classifier(points)
            joints_pred = mask.unsqueeze(-1).expand_as(joints_pred) * joints_pred
            target = mask.unsqueeze(-1).expand_as(target) * target
            
            loss = criterion(cls_logits, label, joints_pred, target, points, mask)

            correct_classes = torch.sum(torch.max(cls_logits, 1)[1] == label).item()
            mean_correct_train.append(correct_classes / float(points.size()[0]))

            sum_mask = mask.sum() * 3
            l1_loss = torch.sum(torch.abs(joints_pred - target)) / sum_mask

            losses_train.append(l1_loss)
            loss.backward()
            optimizer.step()


        with torch.no_grad():

            losses_test = []

            for batch_id, (points, label, mask, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                optimizer.zero_grad()

                # Augmentation
                points = provider.shuffle_points(points)
                points_joint = np.concatenate((points, target), axis=1)
                #points_joint = provider.rotate_point_cloud(points_joint)
                points_joint = provider.random_scale_point_cloud(points_joint)
                points_joint = provider.shift_point_cloud(points_joint)

                points = torch.Tensor(points_joint[:, :points.shape[1], :]) 
                target = torch.Tensor(points_joint[:, points.shape[1]:, :])

                points, label, mask, target = points.cuda().float(), label.cuda().long(), mask.cuda(), target.cuda()
                points = points.transpose(2, 1)
                joints_pred, cls_logits, trans_feat, weights = classifier(points)
                joints_pred = mask.unsqueeze(-1).expand_as(joints_pred) * joints_pred
                target = mask.unsqueeze(-1).expand_as(target) * target

                correct_classes = torch.sum(torch.max(cls_logits, 1)[1] == label).item()
                mean_correct_test.append(correct_classes / float(points.size()[0]))

                sum_mask = mask.sum() * 3
                l1_loss = torch.sum(torch.abs(joints_pred - target)) / sum_mask

                losses_test.append(l1_loss)

                if epoch > 197:
                
                    # Select first example in batch
                    points = points.transpose(2, 1)
                    pc = points[0].cpu().numpy()    # [N, 3]
                    joints = joints_pred[0].cpu().detach().numpy()  # [M, 3]
                    joints = joints[mask[0].cpu().numpy()]  # Apply mask to joints
                    joints_true = target[0].cpu().numpy()  # True joints for comparison
                    joints_true = joints_true[mask[0].cpu().numpy()]  # Apply mask to true joints

                    plot_point_cloud_with_joints(pc, joints, joints_true)

            log_string('Loss train in epoch %d: %f' % (epoch + 1, torch.mean(torch.tensor(losses_train))))
            log_string('Loss test in epoch %d: %f' % (epoch + 1, torch.mean(torch.tensor(losses_test))))
            log_string('Accuracy train in epoch %d: %f' % (epoch + 1, np.mean(mean_correct_train)))
            log_string('Accuracy test in epoch %d: %f' % (epoch + 1, np.mean(mean_correct_test)))


        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
