import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class PointNetPartSeg(nn.Module):
    def __init__(self, num_classes, num_parts):
        super(PointNetPartSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_parts, 1)

        # Classification head
        self.classifier_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, xyz):
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global classification branch
        global_feat = l3_points.view(B, 1024)
        cls_logits = self.classifier_fc(global_feat)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.softmax(x, dim=2)

        # prediction joints
        points_exp = xyz.permute(0, 2, 1).unsqueeze(1)
        weights = x.permute(0, 2, 1).unsqueeze(-1)
        weights = weights.permute(0, 2, 1, 3)
        weighted_sum = torch.sum(weights * points_exp, dim=2)

        return weighted_sum, cls_logits, l3_points, weights


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, cls_logits, labels, pred, target, xyz, mask):

        loss_classes = 0.1 * nn.NLLLoss()(cls_logits, labels)

        sum_mask = mask.sum() * 3
        loss_joint = torch.sum(torch.abs(pred - target)) / sum_mask

        loss = loss_classes + loss_joint

        return loss