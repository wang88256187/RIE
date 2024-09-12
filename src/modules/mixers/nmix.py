import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from egnn_pytorch import EGNN_Sparse, EGNN_Sparse_Network, EGNN_Network, EGNN
# import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt

class Mixer(nn.Module):
    def __init__(self, args, abs=True):
        # torch.save(args, "data_for_evaluate/zerg_args.pth")
        super(Mixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        assert self.qmix_pos_func == "abs"
        
        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

        if self.args.use_graph_state:
            self.gnnbase = GNNBase(args.n_agents, args.n_enemies, args.state_component)
            self.gnnlin = nn.Linear(64, self.input_dim)


        if getattr(args, "use_orthogonal", False):
            raise NotImplementedError
            for m in self.modules():
                orthogonal_init_(m)


    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()
        feats = None
        if self.args.use_graph_state:
            feats = self.gnnbase(states.contiguous().view(b*t,-1))
            states = self.gnnlin(feats).view(b,t,-1)

        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        
        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1, 1)
        
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
        # print(w1.mean(), w1.var())
        # print(w2.mean(), w2.var())

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1) # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        # return y.view(b, t, -1)
        return w1, w2

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)


class GNNBase(nn.Module):
    def __init__(self,
                 n_allies,
                 n_enemies,
                 state_component,
                 max_edge_dist=0.357,
                 egnn_layers=3,
                 gnn_hidden_size=64):
        super().__init__()
        self.max_edge_dist = max_edge_dist
        self.gnn_hidden_size = gnn_hidden_size
        self.n_allies = n_allies
        self.n_enemies = n_enemies
        self.state_component = state_component
        self.action_dim = int(state_component[2] / n_allies)
        self.nf_al = int(state_component[0] / n_allies)
        self.nf_en = int(state_component[1] / n_enemies)
        self.ally_emb = nn.Sequential(nn.Linear(self.nf_al + self.action_dim -2, gnn_hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(gnn_hidden_size, gnn_hidden_size))

        self.enemy_emb = nn.Sequential(nn.Linear(self.nf_en - 2, gnn_hidden_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(gnn_hidden_size, gnn_hidden_size))

        self.egnn = EGNN_Sparse_Network(n_layers=egnn_layers,
                                        feats_dim=gnn_hidden_size,
                                        pos_dim=2)


    def processing_state(self,state):
        ally_feats = state[:, :self.state_component[0]].reshape(-1, self.n_allies, self.nf_al)
        enemy_feats = state[:, self.state_component[0]: self.state_component[0] + self.state_component[1]].\
            reshape(-1, self.n_enemies, self.nf_en)
        last_actions = state[:, self.state_component[0] + self.state_component[1]:].reshape(-1, self.n_allies,self.action_dim)
        ally_feats = torch.cat([ally_feats, last_actions], dim=-1)

        ally_pos = ally_feats[:, :, 2:4]
        ally_other_feats = torch.cat([ally_feats[:, :, :2], ally_feats[:, :, 4:]], dim=-1)
        enemy_pos = enemy_feats[:, :, 1:3]
        enemy_other_feats = torch.cat([enemy_feats[:, :, 0:1], enemy_feats[:,:,3:]], dim=-1)
        entity_pos = torch.cat([ally_pos, enemy_pos], dim=1)

        # find died entities
        ally_died = (ally_feats[:, :, 0] == 0).long()
        enemy_died = (enemy_feats[:, :, 0] == 0).long()
        entity_died = torch.cat([ally_died, enemy_died], dim=1)

        # compute adj for nodes
        entity_pos_y = entity_pos.unsqueeze(1).repeat(1, self.n_allies + self.n_enemies, 1, 1)
        entity_pos_x = entity_pos.unsqueeze(2).repeat(1, 1, self.n_allies + self.n_enemies, 1)
        entity_dist = (entity_pos_x - entity_pos_y).pow(2).sum(-1).sqrt()
        entity_dist[entity_died.unsqueeze(-1).repeat(1,1,20) == 1] = 100
        entity_dist[entity_died.unsqueeze(-2).repeat(1,20,1) == 1] = 100

        adj = (entity_dist < self.max_edge_dist).float()

        # prepare for PyG inputs
        index = adj.nonzero().T
        if len(index) == 3:
            batch = index[0] * adj.size(-1)
            edge_index =torch.stack([batch + index[1], batch + index[2]])

        # index = adj.nonzero(as_tuple=True)
        # if len(index) == 3:
        #     batch = index[0] * adj.size(-1)
        #     edge_index1 = (batch + index[1], batch + index[2])

        ## for debug ##
        # color_dicts = {0: "cyan", 1: "plum"}
        # fig = plt.figure()
        # bs = 1
        # node_x = entity_pos[bs, :, 0]
        # node_y = entity_pos[bs, :, 1]
        # src = entity_pos[bs][edge_index[0]]
        # target = entity_pos[bs][edge_index[1]]
        # colors = [0] * 10 + [1] * 10
        # textCoord = ['0'] * 10 + ['1'] * 10
        # ccc = []
        # # plot nodes
        # for i in colors:
        #     ccc.append(color_dicts[i])
        # plt.scatter(node_x, node_y, s=75, color=ccc)
        # for col, row, t, c in zip(node_x, node_y, textCoord, ccc):
        #     plt.text(col, row, t, fontsize=10, verticalalignment='center', horizontalalignment='center')
        #
        # # plot edges
        # for i in range(len(src)):
        #     plt.plot([src[i][0],target[i][0]], [src[i][1],target[i][1]], color="blue")
        #
        # plt.show()

        return ally_pos, ally_other_feats, enemy_pos, enemy_other_feats, edge_index, entity_died


    def forward(self, state):
        bs = state.size(0)
        ally_pos, ally_other_feats, enemy_pos, enemy_other_feats, edge_index, entity_died = self.processing_state(state)
        ally_other_feats_embedding = self.ally_emb(ally_other_feats)
        enemy_other_feats_embedding = self.enemy_emb(enemy_other_feats)

        ally_node_feats = torch.cat([ally_pos, ally_other_feats_embedding], dim=-1)
        enemy_node_feats = torch.cat([enemy_pos, enemy_other_feats_embedding], dim=-1)

        graph_node_feats = torch.cat([ally_node_feats, enemy_node_feats], dim=1).view(-1, self.gnn_hidden_size+2)
        batch = torch.arange(bs).view(-1, 1).repeat(1, self.n_allies + self.n_enemies).flatten().to(state.device)

        # for debug
        # rotated inputs
        # theta = torch.tensor([torch.pi * 0.5])
        # rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
        #                                 [torch.sin(theta), torch.cos(theta)]])
        # coors = graph_node_feats[:, :2]
        # h = graph_node_feats[:, 2:]
        # rotated_coors = coors @ rotation_matrix
        # rotated_graph_node_feats = torch.cat([rotated_coors, h], dim=-1)
        # rotated_graph_outputs = self.egnn(rotated_graph_node_feats, edge_index, batch)
        # rotated_coors, rotated_feats = rotated_graph_outputs[:, :2], rotated_graph_outputs[:, 2:]

        graph_outputs = self.egnn(graph_node_feats, edge_index, batch, edge_attr=None)
        coors_rep, feats = graph_outputs[:, 0:2], graph_outputs[:, 2:]
        feats = feats.view(bs, self.n_allies+self.n_enemies, -1)
        feats[entity_died == 0] == 0.
        global_feats = feats.max(-2)[0]
        return global_feats


if __name__ == "__main__":
    state = torch.load("test_data/state.pt")[0].reshape(1, -1)
    state1 = torch.load("test_data/state1.pt")[0].reshape(1,-1)
    state = np.concatenate([state,state1], axis=0)
    state = torch.from_numpy(state)

    model = GNNBase(10,10,[80,70,160])
    x, h = model(state)
    print("ok")




