import torch
import torch as th
from torch.distributions import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
import torch.nn as nn
from .epsilon_schedules import DecayThenFlatSchedule
from torch.nn.parameter import Parameter


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class ImpactComputer(nn.Module):
    def __init__(self, own_feats_dim, ally_feats_dim, enemy_feats_dim, total_ob_rep_dim, fc_dim=64):
        super(ImpactComputer, self).__init__()
        ally_input_dim = own_feats_dim + ally_feats_dim + total_ob_rep_dim
        self.ally_impacts_estimater = nn.Sequential(nn.LayerNorm(ally_input_dim),
                                                    init_(nn.Linear(ally_input_dim, fc_dim), activate=True), nn.GELU(),
                                                    nn.LayerNorm(fc_dim),
                                                    init_(nn.Linear(fc_dim, fc_dim), activate=True), nn.GELU(),
                                                    nn.LayerNorm(fc_dim),
                                                    init_(nn.Linear(fc_dim, 1)))

        enemy_input_dim = own_feats_dim + enemy_feats_dim + total_ob_rep_dim
        self.enemy_impacts_estimater = nn.Sequential(nn.LayerNorm(enemy_input_dim),
                                                     init_(nn.Linear(enemy_input_dim, fc_dim), activate=True),
                                                     nn.GELU(), nn.LayerNorm(fc_dim),
                                                     init_(nn.Linear(fc_dim, fc_dim), activate=True), nn.GELU(),
                                                     nn.LayerNorm(fc_dim),
                                                     init_(nn.Linear(fc_dim, 1)))

        self.hyper_w_x = nn.Sequential(nn.Linear(total_ob_rep_dim, fc_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(fc_dim, 2))

        self.hyper_w_y = nn.Sequential(nn.Linear(total_ob_rep_dim, fc_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(fc_dim, 2))

        self.hyper_V = nn.Sequential(nn.Linear(total_ob_rep_dim, fc_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(fc_dim, 1))

    def _compute_resultant_impact(self,
                                  ally_impacts,  ## [bs, n_agents, n_agents-1, 1]
                                  enemy_impacts,
                                  ally_pos,
                                  enemy_pos,
                                  ally_distance,
                                  enemy_distance,
                                  ally_in_sight,
                                  enemy_in_sight):
        # mask those entities information which are not in the agents' sight
        ally_impacts = ally_impacts * ally_in_sight
        enemy_impacts = enemy_impacts * enemy_in_sight

        # ally_distance_clone = ally_distance.clone()
        # enemy_distance_clone = enemy_distance.clone()
        # ally_distance[ally_in_sight == 0] = ally_distance_clone[ally_in_sight == 0] + 1e7
        # enemy_distance[enemy_in_sight == 0] = enemy_distance_clone[enemy_in_sight == 0] + 1e7

        ally_distance[ally_in_sight == 0.] = ally_distance[ally_in_sight == 0.] + 1e7
        enemy_distance[enemy_in_sight == 0.] = enemy_distance[enemy_in_sight == 0.] + 1e7

        # if (ally_distance == 0).any():
        #     print(torch.where(ally_distance == 0))
        ally_cos = ally_pos[:, :, :, 0:1] / ally_distance
        ally_sin = ally_pos[:, :, :, 1:2] / ally_distance
        enemy_cos = enemy_pos[:, :, :, 0:1] / enemy_distance
        enemy_sin = enemy_pos[:, :, :, 1:2] / enemy_distance

        ally_impacts_x = ally_impacts * ally_cos * ally_in_sight
        ally_impacts_y = ally_impacts * ally_sin * ally_in_sight
        enemy_impacts_x = enemy_impacts * enemy_cos * enemy_in_sight
        enemy_impacts_y = enemy_impacts * enemy_sin * enemy_in_sight

        # compute resultant impacts
        resultant_impact_x = ally_impacts_x.sum(2) + enemy_impacts_x.sum(2)
        resultant_impact_y = ally_impacts_y.sum(2) + enemy_impacts_y.sum(2)

        resultant_impact = [resultant_impact_x, resultant_impact_y]

        return resultant_impact

    def forward(self,
                total_ob_rep_for_allys,
                total_ob_rep_for_enemies,
                own_feats_for_allys,
                own_feats_for_enemies,
                ally_feats,
                enemy_feats,
                q_move_mean):
        """
        tensor shape:
        total_ob_rep_for_allys : [bs, n_agents, n_agents - 1, rnn_dim]
        total_ob_rep_for_enemies: [bs, n_agents, n_enemies, rnn_dim]
        own_feats_for_allys: [bs, n_agents, n_agents - 1, own_feats_dim]
        own_feats_for_enemies: [bs, n_agents, n_enemies, own_feats_dim]
        ally_feats:  [bs, n_agents, n_agents - 1, ally_feats_dim]
        enemy_feats: [bs, n_agents, n_enemies, enemies_feats_dim]
        """

        ally_inputs = torch.concat([total_ob_rep_for_allys, own_feats_for_allys, ally_feats], dim=-1)
        enemy_inputs = torch.concat([total_ob_rep_for_enemies, own_feats_for_enemies, enemy_feats], dim=-1)
        ally_impacts = self.ally_impacts_estimater(ally_inputs)  # [bs, n_agents, n_agents - 1, 1]
        enemy_impacts = self.enemy_impacts_estimater(enemy_inputs)  # [bs, n_agents,   n_enemies,  1]

        # entity positions
        ally_pos = ally_feats[:, :, :, 2:4]
        enemy_pos = enemy_feats[:, :, :, 2:4]

        # entity distance
        ally_distance = ally_feats[:, :, :, 1].unsqueeze(-1).clone()
        enemy_distance = enemy_feats[:, :, :, 1].unsqueeze(-1).clone()

        # In it in sight range?
        ally_in_sight = ally_feats[:, :, :, 0].unsqueeze(-1)
        enemy_in_sight = enemy_feats[:, :, :, 0].unsqueeze(-1)

        resultant_impact_x, resultant_impact_y = self._compute_resultant_impact(ally_impacts,
                                                                                enemy_impacts,
                                                                                ally_pos,
                                                                                enemy_pos,
                                                                                ally_distance,
                                                                                enemy_distance,
                                                                                ally_in_sight,
                                                                                enemy_in_sight)

        resultant_impact_1 = torch.concat([resultant_impact_x, -resultant_impact_x], dim=-1)
        resultant_impact_2 = torch.concat([resultant_impact_y, -resultant_impact_y], dim=-1)

        resultant_impact = torch.concat([resultant_impact_y,
                                         -resultant_impact_y,
                                         resultant_impact_x,
                                         -resultant_impact_x], dim=-1)

        q_mean = q_move_mean

        # w_x = self.hyper_w_x(total_ob_rep_for_allys[:, :, 0, :])
        # w_y = self.hyper_w_y(total_ob_rep_for_allys[:, :, 0, :])
        # x_ = resultant_impact_1 * torch.abs(w_x)
        # y_ = resultant_impact_2 * torch.abs(w_y)
        # q_y = q_mean[:, :, 0:2] + y_
        # q_x = q_mean[:, :, 2:4] + x_

        q_y = q_mean[:, :, 0:2] + resultant_impact_2
        q_x = q_mean[:, :, 2:4] + resultant_impact_1
        q_move = torch.concat([q_y, q_x], dim=-1)

        # q_up = q_move_mean + resultant_impact_y
        # q_down = q_move_mean - resultant_impact_y
        # q_right = q_move_mean + resultant_impact_x
        # q_left = q_move_mean - resultant_impact_x

        # q_move = torch.concat([q_up, q_down, q_right, q_left], dim=-1)  # [bs, n_agents,4]

        return q_move


class ImpactComputer22222(nn.Module):
    def __init__(self, own_feats_dim, ally_feats_dim, enemy_feats_dim, total_ob_rep_dim, fc_dim=64):
        super(ImpactComputer22222, self).__init__()
        ally_input_dim = own_feats_dim + ally_feats_dim + total_ob_rep_dim
        self.ally_impacts_estimater = nn.Sequential(nn.LayerNorm(ally_input_dim),
                                                    init_(nn.Linear(ally_input_dim, fc_dim), activate=True), nn.GELU(),
                                                    nn.LayerNorm(fc_dim),
                                                    init_(nn.Linear(fc_dim, fc_dim), activate=True), nn.GELU(),
                                                    nn.LayerNorm(fc_dim),
                                                    init_(nn.Linear(fc_dim, 1)))

        enemy_input_dim = own_feats_dim + enemy_feats_dim + total_ob_rep_dim
        self.enemy_impacts_estimater = nn.Sequential(nn.LayerNorm(enemy_input_dim),
                                                     init_(nn.Linear(enemy_input_dim, fc_dim), activate=True),
                                                     nn.GELU(), nn.LayerNorm(fc_dim),
                                                     init_(nn.Linear(fc_dim, fc_dim), activate=True), nn.GELU(),
                                                     nn.LayerNorm(fc_dim),
                                                     init_(nn.Linear(fc_dim, 1)))

        self.hyper_w_x = nn.Sequential(nn.Linear(total_ob_rep_dim, fc_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(fc_dim, 2))

        self.hyper_w_y = nn.Sequential(nn.Linear(total_ob_rep_dim, fc_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(fc_dim, 2))

        self.hyper_V = nn.Sequential(nn.Linear(total_ob_rep_dim, fc_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(fc_dim, 1))

    def _compute_resultant_impact(self,
                                  ally_impacts,  ## [bs, n_agents, n_agents-1, 1]
                                  enemy_impacts,
                                  ally_pos,
                                  enemy_pos,
                                  ally_distance,
                                  enemy_distance,
                                  ally_in_sight,
                                  enemy_in_sight):
        # mask those entities information which are not in the agents' sight
        ally_impacts = ally_impacts * ally_in_sight
        enemy_impacts = enemy_impacts * enemy_in_sight

        # ally_distance_clone = ally_distance.clone()
        # enemy_distance_clone = enemy_distance.clone()
        # ally_distance[ally_in_sight == 0] = ally_distance_clone[ally_in_sight == 0] + 1e7
        # enemy_distance[enemy_in_sight == 0] = enemy_distance_clone[enemy_in_sight == 0] + 1e7

        ally_distance[ally_in_sight == 0.] = ally_distance[ally_in_sight == 0.] + 1e7
        enemy_distance[enemy_in_sight == 0.] = enemy_distance[enemy_in_sight == 0.] + 1e7

        # if (ally_distance == 0).any():
        #     print(torch.where(ally_distance == 0))
        ally_cos = ally_pos[:, :, :, 0:1] / ally_distance
        ally_sin = ally_pos[:, :, :, 1:2] / ally_distance
        enemy_cos = enemy_pos[:, :, :, 0:1] / enemy_distance
        enemy_sin = enemy_pos[:, :, :, 1:2] / enemy_distance

        ally_impacts_x = ally_impacts * ally_cos * ally_in_sight
        ally_impacts_y = ally_impacts * ally_sin * ally_in_sight
        enemy_impacts_x = enemy_impacts * enemy_cos * enemy_in_sight
        enemy_impacts_y = enemy_impacts * enemy_sin * enemy_in_sight

        # compute resultant impacts
        resultant_impact_x = ally_impacts_x.sum(2) + enemy_impacts_x.sum(2)
        resultant_impact_y = ally_impacts_y.sum(2) + enemy_impacts_y.sum(2)

        resultant_impact = [resultant_impact_x, resultant_impact_y]

        return resultant_impact

    def forward(self,
                total_ob_rep_for_allys,
                total_ob_rep_for_enemies,
                own_feats_for_allys,
                own_feats_for_enemies,
                ally_feats,
                enemy_feats,
                q_move_mean):
        """
        tensor shape:
        total_ob_rep_for_allys : [bs, n_agents, n_agents - 1, rnn_dim]
        total_ob_rep_for_enemies: [bs, n_agents, n_enemies, rnn_dim]
        own_feats_for_allys: [bs, n_agents, n_agents - 1, own_feats_dim]
        own_feats_for_enemies: [bs, n_agents, n_enemies, own_feats_dim]
        ally_feats:  [bs, n_agents, n_agents - 1, ally_feats_dim]
        enemy_feats: [bs, n_agents, n_enemies, enemies_feats_dim]
        """

        ally_inputs = torch.concat([total_ob_rep_for_allys, own_feats_for_allys, ally_feats], dim=-1)
        enemy_inputs = torch.concat([total_ob_rep_for_enemies, own_feats_for_enemies, enemy_feats], dim=-1)
        ally_impacts = self.ally_impacts_estimater(ally_inputs)  # [bs, n_agents, n_agents - 1, 1]
        enemy_impacts = self.enemy_impacts_estimater(enemy_inputs)  # [bs, n_agents,   n_enemies,  1]

        # entity positions
        ally_pos = ally_feats[:, :, :, 2:4]
        enemy_pos = enemy_feats[:, :, :, 2:4]

        # entity distance
        ally_distance = ally_feats[:, :, :, 1].unsqueeze(-1).clone()
        enemy_distance = enemy_feats[:, :, :, 1].unsqueeze(-1).clone()

        # In it in sight range?
        ally_in_sight = ally_feats[:, :, :, 0].unsqueeze(-1)
        enemy_in_sight = enemy_feats[:, :, :, 0].unsqueeze(-1)

        resultant_impact_x, resultant_impact_y = self._compute_resultant_impact(ally_impacts,
                                                                                enemy_impacts,
                                                                                ally_pos,
                                                                                enemy_pos,
                                                                                ally_distance,
                                                                                enemy_distance,
                                                                                ally_in_sight,
                                                                                enemy_in_sight)

        resultant_impact_1 = torch.concat([resultant_impact_x, -resultant_impact_x], dim=-1)
        resultant_impact_2 = torch.concat([resultant_impact_y, -resultant_impact_y], dim=-1)

        resultant_impact = torch.concat([resultant_impact_y,
                                         -resultant_impact_y,
                                         resultant_impact_x,
                                         -resultant_impact_x], dim=-1)

        q_mean = q_move_mean

        # w_x = self.hyper_w_x(total_ob_rep_for_allys[:, :, 0, :])
        # w_y = self.hyper_w_y(total_ob_rep_for_allys[:, :, 0, :])
        # x_ = resultant_impact_1 * torch.abs(w_x)
        # y_ = resultant_impact_2 * torch.abs(w_y)
        # q_y = q_mean[:, :, 0:2] + y_
        # q_x = q_mean[:, :, 2:4] + x_

        q_y = q_mean[:, :, 0:2] + resultant_impact_2
        q_x = q_mean[:, :, 2:4] + resultant_impact_1
        q_move = torch.concat([q_y, q_x], dim=-1)

        # q_up = q_move_mean + resultant_impact_y
        # q_down = q_move_mean - resultant_impact_y
        # q_right = q_move_mean + resultant_impact_x
        # q_left = q_move_mean - resultant_impact_x

        # q_move = torch.concat([q_up, q_down, q_right, q_left], dim=-1)  # [bs, n_agents,4]

        return q_moveclass


class ImpactComputer(nn.Module):
    def __init__(self, own_feats_dim, ally_feats_dim, enemy_feats_dim, total_ob_rep_dim,
                 n_agents, n_enemies, fc_dim=64):
        super(ImpactComputer, self).__init__()
        self.n_agents = n_agents
        self.n_enemies = n_enemies
        self.own_feats_dim = own_feats_dim
        self.ally_feats_dim = ally_feats_dim
        self.enemy_feats_dim = enemy_feats_dim
        self.hidden_dim = total_ob_rep_dim

        ally_input_dim = own_feats_dim + ally_feats_dim + total_ob_rep_dim
        self.ally_impacts_estimater = nn.Sequential(nn.LayerNorm(ally_input_dim),
                                                    init_(nn.Linear(ally_input_dim, fc_dim), activate=True), nn.GELU(),
                                                    nn.LayerNorm(fc_dim),
                                                    init_(nn.Linear(fc_dim, fc_dim), activate=True), nn.GELU(),
                                                    nn.LayerNorm(fc_dim),
                                                    init_(nn.Linear(fc_dim, 1)))

        enemy_input_dim = own_feats_dim + enemy_feats_dim + total_ob_rep_dim
        self.enemy_impacts_estimater = nn.Sequential(nn.LayerNorm(enemy_input_dim),
                                                     init_(nn.Linear(enemy_input_dim, fc_dim), activate=True),
                                                     nn.GELU(), nn.LayerNorm(fc_dim),
                                                     init_(nn.Linear(fc_dim, fc_dim), activate=True), nn.GELU(),
                                                     nn.LayerNorm(fc_dim),
                                                     init_(nn.Linear(fc_dim, 1)))

        # self.hyper_w_x = nn.Sequential(nn.Linear(total_ob_rep_dim, fc_dim),
        #                                nn.ReLU(inplace=True),
        #                                nn.Linear(fc_dim, 2))
        #
        # self.hyper_w_y = nn.Sequential(nn.Linear(total_ob_rep_dim, fc_dim),
        #                                nn.ReLU(inplace=True),
        #                                nn.Linear(fc_dim, 2))
        #
        # self.hyper_V = nn.Sequential(nn.Linear(total_ob_rep_dim, fc_dim),
        #                              nn.ReLU(inplace=True),
        #                              nn.Linear(fc_dim, 1))

    def _compute_resultant_impact(self,
                                  ally_impacts,
                                  enemy_impacts,
                                  ally_cos,
                                  enemy_cos,
                                  ally_sin,
                                  enemy_sin,
                                  ally_in_sight,
                                  enemy_in_sight):
        # mask those entities information which are not in the agents' sight
        # ally_impacts = ally_impacts * ally_in_sight
        # enemy_impacts = enemy_impacts * enemy_in_sight

        # ally_distance_clone = ally_distance.clone()
        # enemy_distance_clone = enemy_distance.clone()
        # ally_distance[ally_in_sight == 0] = ally_distance_clone[ally_in_sight == 0] + 1e7
        # enemy_distance[enemy_in_sight == 0] = enemy_distance_clone[enemy_in_sight == 0] + 1e7

        # ally_distance[ally_in_sight == 0.] = ally_distance[ally_in_sight == 0.] + 1e7
        # enemy_distance[enemy_in_sight == 0.] = enemy_distance[enemy_in_sight == 0.] + 1e7

        # if (ally_distance == 0).any():
        #     print(torch.where(ally_distance == 0))
        # ally_cos = ally_pos[:, :, :, 0:1] / ally_distance
        # ally_sin = ally_pos[:, :, :, 1:2] / ally_distance
        # enemy_cos = enemy_pos[:, :, :, 0:1] / enemy_distance
        # enemy_sin = enemy_pos[:, :, :, 1:2] / enemy_distance

        ally_impacts_x = ally_impacts * ally_cos * ally_in_sight
        ally_impacts_y = ally_impacts * ally_sin * ally_in_sight
        enemy_impacts_x = enemy_impacts * enemy_cos * enemy_in_sight
        enemy_impacts_y = enemy_impacts * enemy_sin * enemy_in_sight

        # compute resultant impacts
        resultant_impact_x = ally_impacts_x.sum(2) + enemy_impacts_x.sum(2)
        resultant_impact_y = ally_impacts_y.sum(2) + enemy_impacts_y.sum(2)

        resultant_impact = [resultant_impact_x, resultant_impact_y]

        return resultant_impact

    def forward(self, ob_rep,
                node_feat,
                ally_in_sight,
                enemy_in_sight,
                ally_cos,
                enemy_cos,
                ally_sin,
                enemy_sin,
                ally_distance,
                enemy_distance,
                q_move_mean):
        """
        tensor shape:
        total_ob_rep_for_allys : [bs, n_agents, n_agents - 1, rnn_dim]
        total_ob_rep_for_enemies: [bs, n_agents, n_enemies, rnn_dim]
        own_feats_for_allys: [bs, n_agents, n_agents - 1, own_feats_dim]
        own_feats_for_enemies: [bs, n_agents, n_enemies, own_feats_dim]
        ally_feats:  [bs, n_agents, n_agents - 1, ally_feats_dim]
        enemy_feats: [bs, n_agents, n_enemies, enemies_feats_dim]
        """

        total_ob_rep_for_allys = ob_rep.reshape(-1, self.n_agents, self.hidden_dim) \
            .unsqueeze(2).expand(-1, -1, self.n_agents - 1, self.hidden_dim)
        total_ob_rep_for_enemies = ob_rep.reshape(-1, self.n_agents, self.hidden_dim) \
            .unsqueeze(2).expand(-1, -1, self.n_enemies, self.hidden_dim)
        own_feats_for_allys = node_feat[:, :, 0].reshape(-1, self.n_agents, self.own_feats_dim) \
            .unsqueeze(2).expand(-1, -1, self.n_agents - 1, -1)
        own_feats_for_enemies = node_feat[:, :, 0].reshape(-1, self.n_agents, self.own_feats_dim) \
            .unsqueeze(2).expand(-1, -1, self.n_enemies, -1)
        enemy_feats = node_feat[:, :, self.n_agents:, :]
        ally_feats = node_feat[:, :, 1:self.n_agents:, :]

        ally_inputs = torch.concat([total_ob_rep_for_allys, own_feats_for_allys, ally_feats], dim=-1)
        enemy_inputs = torch.concat([total_ob_rep_for_enemies, own_feats_for_enemies, enemy_feats], dim=-1)
        ally_impacts = self.ally_impacts_estimater(ally_inputs)  # [bs, n_agents, n_agents - 1, 1]
        enemy_impacts = self.enemy_impacts_estimater(enemy_inputs)  # [bs, n_agents,   n_enemies,  1]

        resultant_impact_x, resultant_impact_y = self._compute_resultant_impact(ally_impacts,
                                                                                enemy_impacts,
                                                                                ally_cos,
                                                                                enemy_cos,
                                                                                ally_sin,
                                                                                enemy_sin,
                                                                                ally_in_sight,
                                                                                enemy_in_sight)

        resultant_impact_1 = torch.concat([resultant_impact_x, -resultant_impact_x], dim=-1)
        resultant_impact_2 = torch.concat([resultant_impact_y, -resultant_impact_y], dim=-1)

        resultant_impact = torch.concat([resultant_impact_y,
                                         -resultant_impact_y,
                                         resultant_impact_x,
                                         -resultant_impact_x], dim=-1)

        # q_mean = q_move_mean

        # q_y = q_mean[:, :, 0:2] + resultant_impact_2
        # q_x = q_mean[:, :, 2:4] + resultant_impact_1
        # q_move = torch.concat([q_y, q_x], dim=-1)
        q_move_mean_ = q_move_mean[:,:,[0,0,1,1]]


        q_move = q_move_mean_ + resultant_impact

        return q_move
