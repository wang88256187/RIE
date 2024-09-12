# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import os
from time import time
import numpy as np


def Qmerge(x):
    y = torch.sum(x * x, dim=-2)
    return y


class QBN_RM1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(QBN_RM1d, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.momentum = momentum
        self.running_average = None

    def forward(self, input):
        xd = input.detach()
        dim_size = xd.size(-1)
        xd2 = xd * xd
        mean2 = xd2.view(-1, dim_size).mean(dim=0)
        mod_xd2 = torch.sqrt(mean2 + self.eps)
        # if not torch.is_tensor(self.running_average):
        #     self.running_average = mod_xd2
        # else:
        #     self.running_average = (1-self.momentum) * self.running_average + self.momentum * mod_xd2
        # coefficient = self.running_average.view(1,inSize[1],1).cuda()
        coefficient = mod_xd2.view(1, 1, -1)
        y = torch.div(input, coefficient)
        return y


def Qrelu1d(x):  # relu activation for quaternion
    xd = x.detach()
    inSize = xd.size()
    threshold = 1.
    mod = torch.sqrt(torch.sum(xd * xd, dim=-2))
    threshold_mod = torch.ones_like(mod) * threshold
    after_thre_mod = torch.max(threshold_mod, mod)
    coefficient = torch.div(mod, after_thre_mod)
    coefficient = coefficient.unsqueeze(-2)
    y = torch.mul(coefficient, x)
    return y


def QMaxPool(x):
    x_detach = x.detach()
    mod = (x_detach * x_detach).sum(dim=-2)
    idx = mod.argmax(-2).unsqueeze(-2).repeat(1, 1, 2, 1).unsqueeze(-3)
    output = torch.gather(x, dim=-3, index=idx)
    output = output.squeeze(-3)

    # for debug
    # max_mod = mod.max(-2)[0]
    # max_mod2 = (output * output).sum(-2)
    return output


class RotatedCovarinceNet(nn.Module):
    def __init__(self, in_channel=1, mlp=[64, 64]):
        super(RotatedCovarinceNet, self).__init__()
        self.mlps = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            # self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, kernel_size=1, bias=False))
            self.mlps.append(nn.Linear(last_channel, out_channel, bias=False))
            self.mlp_bns.append(QBN_RM1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, conv in enumerate(self.mlps):
            bn = self.mlp_bns[i]
            x = conv(x)
            x = bn(x)
            x = Qrelu1d(x)
        # vector_feature = QMaxPool(x)
        return x


class IR_impact(nn.Module):
    def __init(self):
        super(IR_impact, self).__init()
        self.mlp

    pass


class IRAgent(nn.Module):
    def __init__(self):
        super(IRAgent, self).__init__()
        self.MLP_own = nn.Sequential(nn.Linear(11, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU())

        self.RCN_ally = RotatedCovarinceNet(in_channel=19)
        self.RCN_enemy = RotatedCovarinceNet(in_channel=19)

        self.RCN_feat = RotatedCovarinceNet(in_channel=64, mlp=[64, 64])

        self.ir_impact_enemy = RotatedCovarinceNet(in_channel=128, mlp=[64, 64, 1])
        self.ir_impact_ally = RotatedCovarinceNet(in_channel=128, mlp=[64, 64, 1])

        self.MLP_other_action = nn.Sequential(nn.Linear(64, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 3)
                                              )

        self.MLP_attack = nn.Sequential(nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1)
                                        )

    def forward(self, inputs):
        own_feats, enemy_feats, ally_feats = inputs

        own_feats = own_feats.view(-1, 10, 11).cuda()
        enemy_feats = enemy_feats.view(-1, 10, 10, 9).cuda()
        ally_feats = ally_feats.view(-1, 10, 9, 9).cuda()

        own_vector_feats = own_feats.unsqueeze(-2).repeat(1, 1, 2, 1)

        enemy_pos = enemy_feats[:, :, :, 2:4].unsqueeze(-1)
        enemy_pos_mod = (enemy_pos ** 2).sum(-2).sqrt()

        enemy_other_feats_mod = torch.cat([enemy_feats[:, :, :, :2],
                                           enemy_feats[:, :, :, 4:]], dim=-1) \
            .unsqueeze(-2).repeat(1, 1, 1, 2, 1).square().sum(-2).sqrt()
        enemy_coff = (enemy_other_feats_mod / enemy_pos_mod).unsqueeze(-2)
        enemy_coff[enemy_pos_mod.unsqueeze(-2).repeat(1, 1, 1, 1, 7) == 0] = 0
        enemy_vector_feats = enemy_pos.repeat(1, 1, 1, 1, 7) * enemy_coff
        own_for_enemy_mod = own_vector_feats.unsqueeze(-3).repeat(1, 1, 10, 1, 1).square().sum(-2).sqrt()
        own_for_enemy_coff = (own_for_enemy_mod / enemy_pos_mod).unsqueeze(-2)
        own_for_enemy_coff[enemy_pos_mod.unsqueeze(-2).repeat(1, 1, 1, 1, 11) == 0] = 0
        own_for_enemy_feats = enemy_pos.repeat(1, 1, 1, 1, 11) * own_for_enemy_coff
        enemy_vector_feats = torch.cat([enemy_pos,
                                        enemy_vector_feats,
                                        own_for_enemy_feats], dim=-1)

        ally_pos = ally_feats[:, :, :, 2:4].unsqueeze(-1)
        ally_pos_mod = (ally_pos ** 2).sum(-2).sqrt()
        ally_other_feats_mod = torch.cat([ally_feats[:, :, :, :2],
                                          ally_feats[:, :, :, 4:]]
                                         , dim=-1) \
            .unsqueeze(-2).repeat(1, 1, 1, 2, 1).square().sum(-2).sqrt()
        ally_coff = (ally_other_feats_mod / ally_pos_mod).unsqueeze(-2)
        ally_coff[ally_pos_mod.unsqueeze(-2).repeat(1, 1, 1, 1, 7) == 0] = 0
        ally_vector_feats = ally_pos.repeat(1, 1, 1, 1, 7) * ally_coff
        own_for_ally_mod = own_vector_feats.unsqueeze(-3).repeat(1, 1, 9, 1, 1).square().sum(-2).sqrt()
        own_for_ally_coff = (own_for_ally_mod / ally_pos_mod).unsqueeze(-2)
        own_for_ally_coff[ally_pos_mod.unsqueeze(-2).repeat(1, 1, 1, 1, 11) == 0] = 0
        own_for_ally_feats = ally_pos.repeat(1, 1, 1, 1, 11) * own_for_ally_coff

        ally_vector_feats = torch.cat([ally_pos,
                                       ally_vector_feats,
                                       own_for_ally_feats], dim=-1)

        # own_MLP_feats = self.MLP_own(own_feats)
        enemy_RC_feats = self.RCN_enemy(enemy_vector_feats)

        enemy_scalar_feats = Qmerge(enemy_RC_feats)

        ally_RC_feats = self.RCN_ally(ally_vector_feats)

        all_RC_feats = torch.cat([enemy_RC_feats, ally_RC_feats], dim=-3)

        vector_feats = QMaxPool(all_RC_feats)
        vector_feats = self.RCN_feat(vector_feats)

        # move_action_vector = self.RCN_move_action(vector_feats)

        enemy_impact = self.ir_impact_enemy(torch.cat([vector_feats.unsqueeze(-3).repeat(1, 1, 10, 1, 1), enemy_RC_feats],dim=-1))
        ally_impact = self.ir_impact_ally(torch.cat([vector_feats.unsqueeze(-3).repeat(1, 1, 10, 1, 1), enemy_RC_feats],dim=-1))
        move_action_vector = torch.cat([enemy_impact, ally_impact],dim=-3).sum(-3)





        # 添加约束
        # l1 = (move_action_vector[:, :, :, 0] * move_action_vector[:, :, :, 2]).sum(-2).abs()
        # l2 = (move_action_vector[:, :, :, 0] * move_action_vector[:, :, :, 3]).sum(-2).abs()
        # l3 = (move_action_vector[:, :, :, 1] * move_action_vector[:, :, :, 3]).sum(-2).abs()
        # l4 = (move_action_vector[:, :, :, 1] * move_action_vector[:, :, :, 2]).sum(-2).abs()
        #
        # l5 = (move_action_vector[:, :, :, 0] * move_action_vector[:, :, :, 1]).sum(-2)
        # l6 = (move_action_vector[:, :, :, 2] * move_action_vector[:, :, :, 3]).sum(-2)
        #
        # direct_vector = torch.tensor([[0,1],
        #                               [0,-1],
        #                               [1,0],
        #                               [0,1]]).to(move_action_vector.device)
        #
        #
        # p = move_action_vector.unsqueeze(-3).repeat(1,1,4,1,1).permute(0,1,4,2,3)
        #
        # direct_dot_value = (p * direct_vector).sum(-1)
        #
        # move_q, direct = direct_dot_value[:,:,0,:].max(-1)
        # theta = torch.tensor([0, torch.pi, 1.5 * torch.pi, 0.5 * torch.pi])
        #
        # rotation_matrix = torch.tensor([[torch.cos(theta), torch.sin(theta)],
        #                                 [-torch.sin(theta), torch.cos(theta)]]).to(direct.device)
        #
        #
        # new_direct_vector

        # other_action_q = (other_action_vector * other_action_vector).sum(-2)

        scalar_feats = Qmerge(vector_feats)
        other_action_q = self.MLP_other_action(scalar_feats)

        mean_move_q = other_action_q[:, :, 2].unsqueeze(-1)
        up = mean_move_q + move_action_vector[:, :, 1, :]
        down = mean_move_q - move_action_vector[:, :, 1, :]
        right = mean_move_q + move_action_vector[:, :, 0, :]
        left = mean_move_q - move_action_vector[:, :, 0, :]
        move_q = torch.cat([up, down, right, left], dim=-1)

        # attack_action_q = self.MLP_attack(scalar_feats)
        scalar_feats = scalar_feats.unsqueeze(-2).repeat(1, 1, 10, 1)
        attack_action_feats = torch.cat([scalar_feats, enemy_scalar_feats], dim=-1)
        attack_action_q = self.MLP_attack(attack_action_feats).squeeze(-1)
        return torch.cat([other_action_q[:, :, 0:2], move_q, attack_action_q], dim=-1)
        # return torch.cat([other_action_q, attack_action_q], dim=-1), scalar_feats





def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(0)

    inputs = torch.load("ep_batch.pt")
    bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices, inverse_matrix = inputs

    model = IRAgent().cuda()

    ## 验证旋转不变性
    # theta = torch.tensor([torch.pi * 0.5])
    # rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
    #                                 [torch.sin(theta), torch.cos(theta)]])
    #
    # enemy_pos = enemy_feats_t[:,2:4]
    # rotated_enemy_pos = enemy_pos @ rotation_matrix
    # rotated_enemy_feats = enemy_feats_t.clone()
    # rotated_enemy_feats[:,2:4] = rotated_enemy_pos
    #
    # ally_pos = ally_feats_t[:,2:4]
    # rotated_ally_pos = ally_pos @ rotation_matrix
    # rotated_ally_feats = ally_feats_t.clone()
    # rotated_ally_feats[:,2:4] = rotated_ally_pos
    #
    # outputs, feats = model([own_feats_t, enemy_feats_t, ally_feats_t])
    # rotates_outputs, rotated_feats = model([own_feats_t, rotated_enemy_feats, rotated_ally_feats])

    # print("ok")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    mse = nn.MSELoss()
    outputs = model([own_feats_t, enemy_feats_t, ally_feats_t])
    print("outputs:", outputs[0, 0])
    for i in range(1000):
        outputs = model([own_feats_t, enemy_feats_t, ally_feats_t])

        target = torch.ones_like(outputs) * torch.arange(1, 17).cuda()
        loss = mse(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss:", loss.item())

    print("outputs:", outputs[0])

    # enemy_net = RotatedCovarinceNet(in_channel=8).cuda()
    # ally_net = RotatedCovarinceNet(in_channel=8).cuda()
    # own_net = RotatedCovarinceNet(in_channel=11).cuda()
    # own_feats = own_feats_t.unsqueeze(-2).repeat(1,2,1).cuda()
    #
    # enemy_pos = enemy_feats_t[:, 2:4].unsqueeze(-1).cuda()
    # ally_pos = ally_feats_t[:, 2:4].unsqueeze(-1).cuda()
    # enemy_other_feats = torch.cat([enemy_feats_t[:, 0:2], enemy_feats_t[:, 4:]], dim=1)
    # ally_other_feats = torch.cat([ally_feats_t[:, 0:2], ally_feats_t[:, 4:]], dim=1)
    # enemy_other_feats = enemy_other_feats.unsqueeze(-2).repeat(1,2,1).cuda()
    # ally_other_feats = ally_other_feats.unsqueeze(-2).repeat(1,2,1).cuda()
    # enemy_feats = torch.cat([enemy_pos, enemy_other_feats], dim=2)
    # ally_feats = torch.cat([ally_pos, ally_other_feats], dim=2)
    #
    # enemy_output = enemy_net(enemy_feats)
    # ally_output = ally_net(ally_feats)
    #
    # own_output = own_net(own_feats)
    #
    # enemy_output = enemy_output.view(bs,10,-1,2, 64)
    # ally_output = ally_output.view(bs,10,-1,2, 64)
    # own_output = own_output.view(bs,10,-1,2, 64)
    #
    # all_feats = torch.cat([enemy_output, ally_output, own_output], dim=2)
    #
    # output_feats = QMaxPool(all_feats)
    #
    # # 测试旋转不变性
    # theta = torch.tensor([torch.pi * 0.5])
    # rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
    #                                 [torch.sin(theta), torch.cos(theta)]]).cuda()
    #
    # rotated_enemy_pos = rotation_matrix @ enemy_pos
    # rotated_ally_pos = rotation_matrix @ enemy_pos
    #
    # rotated_enemy_feats = rotation_matrix @ enemy_feats
    # rotated_ally_feats = rotation_matrix @ ally_feats
    # rotated_own_feats = rotation_matrix @ own_feats
    #
    # rotated_enemy_output = enemy_net(rotated_enemy_feats).view(bs,10,-1,2, 64)
    # rotated_ally_output = ally_net(rotated_ally_feats).view(bs,10,-1,2, 64)
    # rotated_own_output = own_net(rotated_own_feats).view(bs,10,-1,2, 64)
    #
    # rotated_all_feats = torch.cat([rotated_enemy_output, rotated_ally_output, rotated_own_output], dim=2)
    # rotated_output_feats = QMaxPool(rotated_all_feats)
    #
    # rotated_output_feats_ = rotation_matrix @ output_feats
    print("ok")
