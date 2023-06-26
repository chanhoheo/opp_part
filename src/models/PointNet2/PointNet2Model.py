import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction


class PointNet2_model(nn.Module):
    def __init__(self):
        super(PointNet2_model, self).__init__()
        
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=8, in_channel=256+3, mlp=[256, 512], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=16, in_channel=512+3, mlp=[512, 1024], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024+3, mlp=[1024, 2048, 2048], group_all=True)

        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=8, in_channel=256+3, mlp=[256, 512, 512], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=512, radius=0.4, nsample=16, in_channel=512+3, mlp=[512, 512, 1024], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=128, radius=0.8, nsample=32, in_channel=1024+3, mlp=[1024, 1024], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024+3, mlp=[1024, 2048, 2048], group_all=True)
        
        self.fc1 = nn.Linear(2048, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 1)



    def forward(self, part_desc3d_db, part_kp_xyz):
        """
        Update:
            data (dict): {
                keypoints3d: [N, n1, 3]
                descriptors3d_db: [N, dim, n1]
                scores3d_db: [N, n1, 1]

                query_image: (N, 1, H, W)
                query_image_scale: (N, 2)
                query_image_mask(optional): (N, H, W)
            }
        """
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # part_desc3d_db: [batch, num_part, 7000, 256] -> should be [batch*num_part, 256, 7000]
        # part_kp_xyz: [batch, num_part, 7000, 3] -> should be [batch*num_part, 3, 7000]
        batch = part_desc3d_db.shape[0]
        num_part = part_desc3d_db.shape[1]
        num_point = part_desc3d_db.shape[2]


        feat = part_desc3d_db.view(batch*num_part, num_point, -1)
        feat = feat.permute(0, 2, 1)

        xyz = part_kp_xyz.view(batch*num_part, num_point, -1)
        xyz = xyz.permute(0, 2, 1)


        l1_xyz, l1_points, _ = self.sa1(xyz, feat)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        x = l4_points.view(batch*num_part, -1)

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        x = self.fc3(x) # [batch*num_part, 1]

        x = x.view(batch, num_part) # [batch, num_part]
        x_softmax = F.softmax(x)

        return x, x_softmax