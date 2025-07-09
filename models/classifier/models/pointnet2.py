import torch
import torch.nn as nn
import torch.nn.functional as F
from models.classifier.models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation, add_nearest_joint
from models.classifier.models.gcu import GCU, MLP
from torch_geometric.utils import add_self_loops
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout, InstanceNorm1d
from torch.nn import LayerNorm


class PointNetPartSeg(nn.Module):
    def __init__(self, num_classes: int, num_parts: int, nearest_k: int = 5):
        super().__init__()

        self.nearest_k = nearest_k

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=131, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=259, mlp=[256, 512, 1024], group_all=True)

        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128, 128])

        # Joint prediction head
        self.seg_conv1 = nn.Conv1d(128, 128, kernel_size=1)
        self.seg_bn1 = nn.BatchNorm1d(128)
        self.seg_dropout = nn.Dropout(0.5)
        self.seg_conv2 = nn.Conv1d(128, num_parts, kernel_size=1)

        # Weights prediction layers
        self.encoder_skin = MLP([128, 64, 32, 16])
        self.multi_layer_tranform1 = MLP([3 + 16 + nearest_k*3, 128, 64])
        self.gcu1 = GCU(in_channels=64, out_channels=512, aggr='max')
        self.multi_layer_tranform2 = MLP([512, 512, 1024])
        self.cls_branch = Seq(
            Lin(1536, 1024), ReLU(), LayerNorm(1024), 
            Lin(1024, 512), ReLU(), LayerNorm(512), 
            Lin(512, num_parts)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, xyz: torch.Tensor, tpl_edge_index, mask):
        B, C, N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Classification output
        cls_logits = self._classify(l3_points)

        # Joint position and skinning weights prediction
        joint_pos, skin_weights = self._predict_joints_weights(xyz, l0_points, l1_points, l2_points, l3_points,
                                         l0_xyz, l1_xyz, l2_xyz, l3_xyz, tpl_edge_index, mask)

        return joint_pos, skin_weights, cls_logits

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

    def _predict_weights(self, xyz, feat, joints_pos, tpl_edge_index, mask):

        B, C, N = feat.shape

        # prepare the input
        feat = feat.permute(0, 2, 1)
        feat = self.encoder_skin(feat)
        feat = torch.cat([feat, xyz.permute(0, 2, 1)], dim=2)
        feat = add_nearest_joint(xyz, feat, joints_pos, mask, self.nearest_k)

        x_0 = self.multi_layer_tranform1(feat)
        x_1 = self.gcu1(x_0, tpl_edge_index, tpl_edge_index)
        
        x_global_features = self.multi_layer_tranform2(x_1)
        x_global = x_global_features.max(dim=1, keepdim=True).values
        
        num_nodes = x_1.size(1)
        x_global = x_global.repeat(1, num_nodes, 1)
        x_final = torch.cat([x_1, x_global], dim=2)

        skin_cls_pred = self.cls_branch(x_final)

        return skin_cls_pred

    def _predict_joints_weights(self, xyz, l0_points, l1_points, l2_points, l3_points,
                        l0_xyz, l1_xyz, l2_xyz, l3_xyz, tpl_edge_index, mask) -> torch.Tensor:

        # Decode features
        feat = self._decode_features(l0_xyz, l1_xyz, l2_xyz, l3_xyz,
                                     l0_points, l1_points, l2_points, l3_points)

        # predict joint positions
        feat_joint = self.seg_dropout(feat)
        feat_joint = self.seg_conv2(feat_joint)
        joints_pos = F.softmax(feat_joint, dim=2)
        joints_pos = torch.einsum('bcn,bjn->bjc', xyz, joints_pos)

        # predict skinning weights
        skin_logits = self._predict_weights(xyz, feat, joints_pos, tpl_edge_index, mask)

        return joints_pos, skin_logits


class SkinningLoss(nn.Module):
    def __init__(self, class_weight=0.1, skin_weight=0.1):
        super().__init__()
        self.class_weight = class_weight
        self.skin_weight = skin_weight

    def forward(self, cls_logits, labels, pred, target, mask, skin_weights, weights_gt):
        B, N, P = skin_weights.shape

        # Class Loss
        class_loss = nn.CrossEntropyLoss()(cls_logits, labels)

        # Joint Coordinate Loss (L1)
        l1_loss = F.l1_loss(pred, target, reduction='none') * mask.unsqueeze(-1)
        joint_loss = l1_loss.sum() / (mask.sum() * pred.shape[-1])

        # Skin Loss
        skin_loss = 0.0
        for i in range(B):
            skin_weight_current = skin_weights[i, :, :]
            skin_weight_gt_current = weights_gt[i, :, :]

            skin_weight_current = skin_weight_current[:, mask[i]]
            skin_weight_gt_current = skin_weight_gt_current[:, mask[i]]

            skin_log_probs = F.log_softmax(skin_weight_current, dim=1)
            skin_loss_current = -skin_weight_gt_current * skin_log_probs
            skin_loss += skin_loss_current.sum(dim=1).mean()
        skin_loss /= B

        loss = (self.class_weight * class_loss) + (self.skin_weight * skin_loss) + joint_loss

        return loss, {
            'class_loss': class_loss.item(),
            'joint_loss': joint_loss.item(),
            'skin_loss': skin_loss.item()
        }
