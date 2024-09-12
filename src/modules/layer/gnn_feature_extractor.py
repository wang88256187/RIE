import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    NNConv,
    global_mean_pool,
    graclus,
    max_pool,
    max_pool_x,
)

from torch_geometric.utils import normalized_cut

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, NNConv
from torch_geometric.nn.conv import NNConv
import networkx as nx
import matplotlib.pyplot as plt

num_features = 9
n_agent = 10
bs = 1
n_entity = 20

transform = T.ToUndirected


def visualize_graph(G, pos):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=pos, with_labels=False,
                     cmap="Set2")
    plt.show()


def normalized_cut_2d(edge_index, edge_attr, n_nodes):
    edge_attr1 = edge_attr[:, 0]
    return normalized_cut(edge_index, edge_attr1, num_nodes=n_nodes)


def data_processing(node_feats, adj, edge_feat, bs, n_agent, n_entity, agents_feat_dim):
    data_list = []
    node_feats = node_feats.reshape(bs * n_agent, n_entity, agents_feat_dim)
    adj = adj.reshape(bs * n_agent, n_entity, n_entity)
    edge_feat = edge_feat.reshape(bs * n_agent, n_entity, n_entity, 1)
    for i in range(node_feats.shape[0]):
        x = node_feats[i]
        edge_index = adj[i].to_sparse_coo().indices()
        edge_attr = edge_feat[i][edge_index[0], edge_index[1]]
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    loader = DataLoader(data_list, batch_size=bs * n_agent)
    data = iter(loader).__next__()
    return data

def data_processing3(node_feats, adj, edge_feat, bs, n_agent, n_entity, agents_feat_dim):
    data_list = []
    node_feats = node_feats.reshape(bs * n_agent, n_entity, agents_feat_dim)
    adj = adj.reshape(bs * n_agent, n_entity, n_entity)
    edge_feat = edge_feat.reshape(bs * n_agent, n_entity, n_entity, 1)
    batch = torch.arange(bs * n_agent, device=edge_indexs.device).reshape(-1,1).repeat(1, n_entity).reshape
    sparse_adj = torch.zeros((bs * n_agent * n_entity, bs * n_agent * n_entity))
    sparse_adj[:, ]

    for i in range(node_feats.shape[0]):
        x = node_feats[i]
        edge_index = adj[i].to_sparse_coo().indices()
        edge_attr = edge_feat[i][edge_index[0], edge_index[1]]
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    loader = DataLoader(data_list, batch_size=bs * n_agent)
    data = iter(loader).__next__()
    return data


def data_processing2(node_feats, entity_between_distance, entity_clockwise_edge, n_valid_id,
                     entity_in_sight, entity_distance, bs, n_agent, n_entity, feat_dim):
    node_feats = node_feats.reshape(bs * n_agent, n_entity, feat_dim)
    entity_clockwise_edge = entity_clockwise_edge.reshape(bs * n_agent, n_entity-1, 2)
    entity_between_distance = entity_between_distance.reshape(bs * n_agent, n_entity-1)
    entity_in_sight = entity_in_sight.reshape(bs * n_agent, -1)
    n_valid_id = n_valid_id.reshape(bs * n_agent, 1)
    entity_distance = entity_distance.reshape(bs * n_agent, -1)
    edge_indexs = []
    edge_attrs = []
    for i in range(node_feats.shape[0]):
        # x = node_feats[i]
        target_id = torch.where(entity_in_sight[i] == 1)[0]
        source_id = torch.zeros_like(target_id)
        agents_edges = torch.vstack([source_id[None], target_id[None] + 1]).T
        agents_edges_ = torch.vstack([target_id[None] + 1, source_id[None]]).T
        round_edges = entity_clockwise_edge[i][:n_valid_id[i]] + 1
        round_edges_ = entity_clockwise_edge[i][:n_valid_id[i]][:, [1,0]]
        entity_self_loop_edges = torch.vstack([target_id[None] + 1, target_id[None] + 1]).T
        agents_self_loop_edge = torch.zeros((1, 2), device=node_feats.device)
        edge_index = torch.vstack([agents_self_loop_edge,
                                   entity_self_loop_edges,
                                   agents_edges,
                                   agents_edges_,
                                   round_edges,
                                   round_edges_]).T.long()
        # print(edge_index.shape[1])
        entity_self_loop_attr = torch.zeros_like(target_id)
        agents_self_loop_attr = torch.zeros([1], device=node_feats.device)
        round_edge_attr = entity_between_distance[i][:n_valid_id[i]]
        agents_edges_attr = entity_distance[i][target_id]
        edge_attr = torch.cat([agents_self_loop_attr,
                               entity_self_loop_attr,
                               agents_edges_attr,
                               agents_edges_attr,
                               round_edge_attr,
                               round_edge_attr])
        edge_indexs.append(edge_index + n_entity * i)
        edge_attrs.append(edge_attr)

        # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # data = T.ToUndirected()(data)
        # data = T.AddSelfLoops("edge_weight",1.0)(data)
        # data_list.append(data)

    # loader = DataLoader(data_list, batch_size=bs * n_agent)
    # data = iter(loader).__next__()
    # data = T.ToUndirected()(data)
    edge_indexs = torch.hstack(edge_indexs)
    edge_attrs = torch.cat(edge_attrs).reshape(-1,1)
    batch = torch.arange(bs * n_agent, device=edge_indexs.device).reshape(-1,1).repeat(1, n_entity).flatten()
    return node_feats.reshape(-1, feat_dim), edge_indexs, edge_attrs, batch


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(1, 16), nn.ReLU(),
                            nn.Linear(16, num_features * 32))
        self.conv1 = NNConv(num_features, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(1, 16), nn.ReLU(),
                            nn.Linear(16, 32 * 64))
        self.conv2 = NNConv(32, 64, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 64)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.elu(self.conv1(x, edge_index, edge_attr))

        node_feat = F.elu(self.conv2(x, edge_index, edge_attr))

        x = global_mean_pool(node_feat, batch)
        x = F.elu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        return self.fc2(x), node_feat


if __name__ == "__main__":
    model = GCN().cuda()
    print(model)

    node_feats, adj, edge_feat, ally_in_sight, \
    enemy_in_sight, ally_cos, enemy_cos, ally_sin, \
    enemy_sin, ally_distance, enemy_distance, \
    entity_between_distance, entity_clockwise_edge, \
    n_valid_id, entity_in_sight, entity_distance = torch.load("example_data.pt")

    x, edge_index, edge_attr, batch = data_processing2(node_feats, entity_between_distance, entity_clockwise_edge, n_valid_id, entity_in_sight, entity_distance, 1, 10, 20, 9)

    # 可视化



    data = data_processing(node_feats, adj, edge_feat, 1, 10, 20, 9)



    # xx = data.x.cpu().numpy()
    gobal_feats, entity_feats = model(data.x, data.edge_index, data.edge_attr, data.batch)


    gobal_feats2, entity_feats2 = model(x, edge_index, edge_attr, batch)

    print(gobal_feats.shape)

    # 可视化
    # num_nodes = 20  # 需要可视化的节点数
    # graph = nx.Graph()  # 创建一个图
    #
    # x1 = x[20:40]
    # edge_index1 = edge_index[:, 46:96]
    # for i in range(46):
    #     graph.add_edge(edge_index1[0][i].item() - 20, edge_index1[1][i].item() - 20)
    #
    # pos = nx.kamada_kawai_layout(graph)
    # nx.draw(graph, pos, node_size=50)
    # plt.savefig("2")
    #
    # num_nodes = 20  # 需要可视化的节点数
    # graph2 = nx.Graph()  # 创建一个图
    #
    # x2 = data.x[20:40]
    # edge_index2 = data.edge_index[:, 46:92]
    # for i in range(46):
    #     graph.add_edge(edge_index2[0][i].item(), edge_index2[1][i].item())
    #
    # pos = nx.kamada_kawai_layout(graph)
    # nx.draw(graph2, pos, node_size=50)
    # plt.savefig("1")


# data_list = []
# node_feats = node_feats.reshape(bs * n_agent, n_entity, 9)
# adj = adj.reshape(bs * n_agent, n_entity, n_entity)
# edge_feat = edge_feat.reshape(bs * n_agent, n_entity, n_entity,1)
#
# for i in range(node_feats.shape[0]):
#     x = node_feats[i]
#     edge_index = adj[i].to_sparse_coo().indices()
#     edge_attr = edge_feat[i][edge_index[0], edge_index[1]]
#     data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
#
# loader = DataLoader(data_list, batch_size=bs * n_agent)


# agent_id = 2
# G = adj[0, agent_id, :, :, 0].cpu().numpy()
# sparse_G = adj.reshape(1 * 10, 20, 20).to_sparse_coo().indices()
# x = node_feats.reshape(1 * 10 * 20, -1)
# output = model(x, sparse_G)

# node_pos = entity_pos[0, agent_id].cpu()
# self_pos = torch.tensor([[0, 0]])
# pos = torch.concat([self_pos, node_pos]).numpy()
# nx_G = nx.from_numpy_matrix(G)
# a = nx_G.get_edges()
# edge_index = torch.tensor(nx_G.edges).cuda()
# output = model(x, edge_index)
#
# print(nx_G)
# print(nx_G.edges)

# visualize_graph(nx_G, pos)
