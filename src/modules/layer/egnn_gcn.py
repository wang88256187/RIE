import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from egnn_pytorch import EGNN_Sparse, EGNN_Sparse_Network, EGNN_Network, EGNN

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class IRAgent(nn.Module):
    def __init__(self, own_feats_dim, ally_feats_dim, enemy_feats_dim, hidden_dim=64, dims=[64, 64], ):
        super().__init__()



        self.fc_own = nn.Linear(own_feats_dim-2, hidden_dim)
        self.fc_ally = nn.Linear(ally_feats_dim-2, hidden_dim)
        self.fc_enemy = nn.Linear(enemy_feats_dim-2, hidden_dim)

        self.egnnbase = nn.ModuleList()

        for feat_dim in dims:
            self.egnnbase.append(EGNN_Sparse(feats_dim=feat_dim, pos_dim=2))

    def forward(self, inputs):
        bs, own_feats, ally_feats, enemy_feats = inputs



        pass





if __name__ == "__main__":
    setup_seed(0)
    feats = torch.randn(16, 16)
    coors = torch.randn(16, 2)
    adj = torch.from_numpy(np.random.choice([0, 1], 256, [0.8, 0.2])).view(16, 16)
    adj_sparse = adj.to_sparse()
    edge_index = adj_sparse.indices()
    batch = torch.zeros(16).long()
    edge_attr = adj_sparse.values()[:, None]

    # layer = EGNN_Sparse(feats_dim=16, pos_dim=2)
    # outputs = layer(x, edge_index=edge_index, batch=batch)

    model = EGNN_Sparse_Network(n_layers=2,
                                feats_dim=16,
                                pos_dim=2,
                                edge_attr_dim=1,
                                )

    # original inputs
    x = torch.cat([coors, feats], dim=-1)

    # rotated inputs
    theta = torch.tensor([torch.pi * 0.5])
    rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                    [torch.sin(theta), torch.cos(theta)]])

    rotated_coors = coors @ rotation_matrix

    rotated_x = torch.cat([rotated_coors, feats], dim=-1)


    output1 = model(x, edge_index, batch, edge_attr)
    output1_x = output1[:,0:2]
    output1_h = output1[:,2:]

    output2 = model(rotated_x, edge_index, batch, edge_attr)
    output2_x = output2[:, 0:2]
    output2_h = output2[:, 2:]

    x3 = output1_x @ rotation_matrix

    print("ok")