import torch
import torch.nn as nn
import torch.nn.functional as F
from models.classifier.models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class PointNetPartSeg(nn.Module):
    def __init__(self, num_classes: int, num_parts: int):
        super().__init__()

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=259, mlp=[256, 512, 1024], group_all=True)

        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128, 128])

        # Segmentation head
        self.seg_conv1 = nn.Conv1d(128, 128, kernel_size=1)
        self.seg_bn1 = nn.BatchNorm1d(128)
        self.seg_dropout = nn.Dropout(0.5)
        self.seg_conv2 = nn.Conv1d(128, num_parts, kernel_size=1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, xyz: torch.Tensor):
        B, C, N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Classification output
        cls_logits = self._classify(l3_points)

        # Joint prediction
        joint_pos = self._predict_joints(xyz, l0_points, l1_points, l2_points, l3_points,
                                         l0_xyz, l1_xyz, l2_xyz, l3_xyz)

        return joint_pos, cls_logits

    def _classify(self, global_feat: torch.Tensor) -> torch.Tensor:
        x = global_feat.view(global_feat.size(0), -1)
        return self.classifier(x)

    def _decode_features(self, l0_xyz, l1_xyz, l2_xyz, l3_xyz,
                         l0_points, l1_points, l2_points, l3_points) -> torch.Tensor:
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_input = torch.cat([l0_xyz, l0_points], dim=1)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_input, l1_points)

        x = self.seg_conv1(l0_points)
        x = self.seg_bn1(x)
        return F.relu(x)

    def _predict_joints(self, xyz, l0_points, l1_points, l2_points, l3_points,
                        l0_xyz, l1_xyz, l2_xyz, l3_xyz) -> torch.Tensor:
        feat = self._decode_features(l0_xyz, l1_xyz, l2_xyz, l3_xyz,
                                     l0_points, l1_points, l2_points, l3_points)

        weights = self.seg_dropout(feat)
        weights = self.seg_conv2(weights)
        weights = F.softmax(weights, dim=2)

        # Weighted joint prediction
        return torch.einsum('bcn,bjn->bjc', xyz, weights)


class JointPredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_loss_fn = nn.NLLLoss()

    def forward(self, cls_logits, labels, pred, target, mask):
        class_loss = 0.1 * self.class_loss_fn(cls_logits, labels)
        joint_loss = F.l1_loss(pred, target, reduction='sum') / (mask.sum() * 3)
        return class_loss + joint_loss
