import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from modules.layer.rotate_invarance_net import RotatedCovarinceNet


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
        mean2 = xd2.view(-1, dim_size).mean(dim=0).cuda()
        mod_xd2 = torch.sqrt(mean2 + self.eps)
        # if not torch.is_tensor(self.running_average):
        #     self.running_average = mod_xd2
        # else:
        #     self.running_average = (1-self.momentum) * self.running_average + self.momentum * mod_xd2
        # coefficient = self.running_average.view(1,inSize[1],1).cuda()
        coefficient = mod_xd2.view(1, 1, -1).cuda()
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
    dims = len(x.size())

    x_detach = x.detach()
    mod = (x_detach * x_detach).sum(dim=-2)
    idx = mod.argmax(-2).unsqueeze(-2).repeat(1, 1, 2, 1).unsqueeze(-3)
    output = torch.gather(x, dim=-3, index=idx)
    output = output.squeeze(-3)

    # for debug
    # max_mod = mod.max(-2)[0]
    # max_mod2 = (output * output).sum(-2)
    return output


class IRAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(IRAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = self.n_agents - 1
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim

        # [4 + 1, (6, 5), (4, 5)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]

        self.MLP_own = nn.Sequential(nn.Linear(11, self.rnn_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim),
                                     nn.ReLU())

        self.RCN_ally = RotatedCovarinceNet(in_channel=self.ally_feats_dim + self.own_feats_dim -1)
        self.RCN_enemy = RotatedCovarinceNet(in_channel=self.enemy_feats_dim + self.own_feats_dim -1)

        self.RCN_feat = RotatedCovarinceNet(in_channel=self.rnn_hidden_dim, mlp=[64, 64])

        # self.RCN_other_action = RotatedCovarinceNet(in_channel=64, mlp=[64, 6])

        # self.RCN_move_action = RotatedCovarinceNet(in_channel=64, mlp=[64, 64, 1])
        self.ir_impact_enemy = RotatedCovarinceNet(in_channel=128, mlp=[64, 64, 1])
        self.ir_impact_ally = RotatedCovarinceNet(in_channel=128, mlp=[64, 64, 1])


        self.MLP_attack = nn.Sequential(nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 1)
                                        )

        self.MLP_other_action = nn.Sequential(nn.Linear(64, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 3)
                                              )

        # self.MLP_attack = nn.Sequential(nn.Linear(64, 128),
        #                                 nn.ReLU(),
        #                                 nn.Linear(128, 64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64, 10)
        #                                 )



        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)



        if self.args.map_type in ["MMM", "terran_gen"]:
            assert self.n_enemies >= self.n_agents, "For MMM map, for the reason that the 'attack' and 'rescue' use the same ids in SMAC, n_enemies must >= n_agents"
            self.hyper_output_w_rescue_action = Hypernet(
                input_dim=self.ally_feats_dim, hidden_dim=args.hpn_hyper_dim,
                main_input_dim=self.rnn_hidden_dim, main_output_dim=1,
                activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
            )  # output shape: rnn_hidden_dim * 1
            self.hyper_output_b_rescue_action = Hypernet(
                input_dim=self.ally_feats_dim, hidden_dim=args.hpn_hyper_dim,
                main_input_dim=1, main_output_dim=1,
                activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
            )  # output shape: 1
            self.unify_rescue_output_heads = Merger(self.n_heads, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        # return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()
        return None

    def forward(self, inputs, hidden_state):
        """
               own_feats:     move_feats(4)，
                              相对血量(1)，
                              【相对护盾值(1位)】(神族专属)，
                              相对位置pos(2位)，
                              兵种id(3位)

               enemy_feats:   能否被攻击(1位)，
                              相对距离(1位)，
                              相对位置pos(2位)，
                              相对血量(1位)，
                              【相对护盾值(1位)】(神族专属)
                              兵种id(3位)

               ally_feats:   能否被看见(1),
                             相对距离(1位)，
                             相对位置pos(2位)，
                             相对血量(1位)，
                             【相对护盾值(1位)】(神族专属)
                             兵种id(3位)

               """




        # [bs * n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim]
        bs, own_feats, enemy_feats, ally_feats, embedding_indices, inverse_matrix = inputs


        own_feats = own_feats.view(-1, 10, 11)
        enemy_feats = enemy_feats.view(-1, 10, 10, 9)
        ally_feats = ally_feats.view(-1, 10, 9, 9)

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

        # other_action_vector = self.RCN_other_action(vector_feats)
        # other_action_q = (other_action_vector * other_action_vector).sum(-2)

        # move_action_vector = self.RCN_move_action(vector_feats)

        enemy_impact = self.ir_impact_enemy(
            torch.cat([vector_feats.unsqueeze(-3).repeat(1, 1, 10, 1, 1), enemy_RC_feats], dim=-1))
        ally_impact = self.ir_impact_ally(
            torch.cat([vector_feats.unsqueeze(-3).repeat(1, 1, 10, 1, 1), enemy_RC_feats], dim=-1))
        move_action_vector = torch.cat([enemy_impact, ally_impact], dim=-3).sum(-3)


        scalar_feats = Qmerge(vector_feats)
        other_action_q = self.MLP_other_action(scalar_feats)

        mean_move_q = other_action_q[:, :, 2].unsqueeze(-1)
        up = mean_move_q + move_action_vector[:, :, 1, :]
        down = mean_move_q - move_action_vector[:, :, 1, :]
        right = mean_move_q + move_action_vector[:, :, 0, :]
        left = mean_move_q - move_action_vector[:, :, 0, :]
        move_q = torch.cat([up, down, right, left], dim=-1)

        scalar_feats = scalar_feats.unsqueeze(-2).repeat(1, 1, 10, 1)
        attack_action_feats = torch.cat([scalar_feats, enemy_scalar_feats], dim=-1)
        attack_action_q = self.MLP_attack(attack_action_feats).squeeze(-1)

        # other_action_q = self.MLP_other_action(scalar_feats)
        # attack_action_q = self.MLP_attack(scalar_feats)

        return torch.cat([other_action_q[:,:,0:2], move_q, attack_action_q], dim=-1), None








