#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch

from components.episode_buffer import EpisodeBatch
import copy
import numpy as np
import torch as th
import matplotlib.pyplot as plt


def clear_no_reward_sub_trajectory(batch):
    """
    :param batch:
    :return:
    """
    filled = batch.data.transition_data["filled"]  # [bs, traj_length, 1]
    rewards = batch.data.transition_data["reward"]  # [bs, traj_length, 1]
    bs, traj_length = filled.shape[0], filled.shape[1]
    fixed_row = []
    for t in range(traj_length - 1, 0, -1):
        remained_rows = [i for i in range(0, bs) if i not in fixed_row]
        for row_idx in remained_rows:
            if rewards[row_idx, t - 1, 0] == 0:  # no reward
                filled[row_idx, t, 0] = 0
                if t == 1:
                    filled[row_idx, t - 1, 0] = 0  # the trajectory's Return is 0.
            else:  # receive reward
                fixed_row.append(row_idx)

    return batch[fixed_row]


def _get_obs_component_dim(args):
    move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = args.obs_component  # [4, (6, 5), (4, 5), 1]
    enemy_feats_dim = np.prod(enemy_feats_dim)
    ally_feats_dim = np.prod(ally_feats_dim)
    return move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim


def _generate_permutation_matrix(bs, seq_length, n_agents, N, device):
    permutation_matrix = th.zeros(size=[bs, seq_length, n_agents, N, N], dtype=th.float32, device=device)
    ordered_list = np.arange(N)  # [0, 1, 2, 3, ...]
    shuffled_list = ordered_list.copy()
    np.random.shuffle(shuffled_list)  # [3, 0, 2, 1, ...]
    permutation_matrix[:, :, :, ordered_list, shuffled_list] = 1
    return permutation_matrix


def do_data_augmentation(args, batch: EpisodeBatch, augment_times=2):
    """
    'obs', 'attack action' and 'available action' need to be transformed
    :param args:
    :param batch:
    :param augment_times:
    :return:
    """
    bs = batch.batch_size
    seq_length = batch.max_seq_length
    obs_component_dim = _get_obs_component_dim(args=args)
    attack_action_start_idx = 6

    augmented_data = []
    for t in range(augment_times):
        new_batch = copy.deepcopy(batch)
        obs = new_batch.data.transition_data["obs"]  # [bs, seq_length, n_agents, obs_dim]
        # actions = new_batch.data.transition_data["actions"]  # [bs, seq_length, n_agents, 1]
        actions_onehot = new_batch.data.transition_data["actions_onehot"]  # [bs, seq_length, n_agents, action_num]
        avail_actions = new_batch.data.transition_data["avail_actions"]  # [bs, seq_length, n_agents, action_num]

        # (1) split observation according to the semantic meaning
        move_feats, enemy_feats, ally_feats, own_feats = th.split(obs, obs_component_dim, dim=-1)
        reshaped_enemy_feats = enemy_feats.contiguous().view(bs, seq_length, args.n_agents, args.n_enemies, -1)
        reshaped_ally_feats = ally_feats.contiguous().view(bs, seq_length, args.n_agents, (args.n_agents - 1), -1)

        # (2) split available action into 2 groups: 'move' and 'attack'.
        avail_other_action = avail_actions[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
        avail_attack_action = avail_actions[:, :, :, attack_action_start_idx:]  # [n_enemies]

        # (3) split actions_onehot into 2 groups: 'move' and 'attack'.
        other_action_onehot = actions_onehot[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
        attack_action_onehot = actions_onehot[:, :, :, attack_action_start_idx:]  # [n_enemies]

        # (4) generate permutation matrix for 'ally' and 'enemy'
        ally_perm_matrix = _generate_permutation_matrix(bs, seq_length, args.n_agents, args.n_agents - 1,
                                                        device=obs.device)
        enemy_perm_matrix = _generate_permutation_matrix(bs, seq_length, args.n_agents, args.n_enemies,
                                                         device=obs.device)

        # (5) permute obs: including ally and enemy
        # [bs, seq_length, n_agents, N, N] * [bs, seq_length, n_agents, N, feature_dim]
        permuted_enemy_feat = th.matmul(enemy_perm_matrix, reshaped_enemy_feats).view(bs, seq_length, args.n_agents, -1)
        permuted_ally_feat = th.matmul(ally_perm_matrix, reshaped_ally_feats).view(bs, seq_length, args.n_agents, -1)
        permuted_obs = th.cat([move_feats, permuted_enemy_feat, permuted_ally_feat, own_feats], dim=-1)
        # permuted_obs = th.cat([move_feats, permuted_enemy_feat, ally_feats, own_feats], dim=-1)

        # (6) permute available action (use the same permutation matrix for enemy)
        permuted_avail_attack_action = th.matmul(enemy_perm_matrix, avail_attack_action.unsqueeze(-1).float()).view(
            bs, seq_length, args.n_agents, -1)
        permuted_avail_actions = th.cat([avail_other_action, permuted_avail_attack_action.int()], dim=-1)

        # (7) permute attack_action_onehot (use the same permutation matrix for enemy)
        #     used when obs_last_action is True
        permuted_attack_action_onehot = th.matmul(enemy_perm_matrix, attack_action_onehot.unsqueeze(-1).float()).view(
            bs, seq_length, args.n_agents, -1)
        permuted_action_onehot = th.cat([other_action_onehot, permuted_attack_action_onehot], dim=-1)
        permuted_action = permuted_action_onehot.max(dim=-1, keepdim=True)[1]

        new_batch.data.transition_data["obs"] = permuted_obs
        new_batch.data.transition_data["actions"] = permuted_action
        new_batch.data.transition_data["actions_onehot"] = permuted_action_onehot
        new_batch.data.transition_data["avail_actions"] = permuted_avail_actions

        if augment_times > 1:
            augmented_data.append(new_batch)
    if augment_times > 1:
        return augmented_data
    return new_batch


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


def visualize_pos(ori_pos, rotated_pos, theta):
    fig = plt.figure(figsize=(24, 10))

    fig.suptitle("rotation angle is %.2f:" % (theta / (2 * np.pi) * 360))
    ax1 = plt.subplot(1, 2, 1)
    plt.scatter(ori_pos[:, 0], ori_pos[:, 1])
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.scatter(rotated_pos[:, 0], rotated_pos[:, 1])
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)

    plt.show()


def data_random_rotate(args, batch: EpisodeBatch, theta):
    # rotation_angles = th.rand(augment_times)

    bs = batch.batch_size
    seq_length = batch.max_seq_length
    obs_component_dim = _get_obs_component_dim(args=args)
    attack_action_start_idx = 6
    augmented_data = []
    # for t in range(rotate_times):
    new_batch = copy.deepcopy(batch)
    obs = new_batch.data.transition_data["obs"]  # [bs, seq_length, n_agents, obs_dim]
    # actions = new_batch.data.transition_data["actions"]  # [bs, seq_length, n_agents, 1]
    actions_onehot = new_batch.data.transition_data["actions_onehot"]  # [bs, seq_length, n_agents, action_num]
    avail_actions = new_batch.data.transition_data["avail_actions"]  # [bs, seq_length, n_agents, action_num]
    # states = new_batch.data.transition_data["avail_actions"]
    # (1) split observation according to the semantic meaning
    move_feats, enemy_feats, ally_feats, own_feats = th.split(obs, obs_component_dim, dim=-1)
    reshaped_enemy_feats = enemy_feats.contiguous().view(bs, seq_length, args.n_agents, args.n_enemies, -1)
    reshaped_ally_feats = ally_feats.contiguous().view(bs, seq_length, args.n_agents, (args.n_agents - 1), -1)

    # (2) split available action into 2 groups: 'move' and 'attack'.
    avail_other_action = avail_actions[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
    avail_attack_action = avail_actions[:, :, :, attack_action_start_idx:]  # [n_enemies]

    # (3) split actions_onehot into 2 groups: 'move' and 'attack'.
    other_action_onehot = actions_onehot[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
    attack_action_onehot = actions_onehot[:, :, :, attack_action_start_idx:]  # [n_enemies]

    # (4) generate rotation matrix
    # theta = 2 * th.rand(bs) * torch.pi

    # theta = 1.0 * th.pi * th.ones(bs)

    enemy_pos_rotation_matrix = torch.stack(
        [torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)], dim=1). \
        view(-1, 1, 1, 1, 2, 2).repeat(1, seq_length, args.n_agents, args.n_enemies, 1, 1).cuda()
    ally_pos_rotation_matrix = torch.stack(
        [torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)], dim=1). \
        view(-1, 1, 1, 1, 2, 2).repeat(1, seq_length, args.n_agents, args.n_agents-1, 1, 1).cuda()
    # ally_pos_rotation_matrix = torch.cat(
    #         [torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)]). \
    #         view(-1, 2, 2).cuda()
    action_rotation_matrix = torch.stack([torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)],
                                         dim=1). \
        view(-1, 1, 1, 2, 2).repeat(1, seq_length, args.n_agents, 1,
                                    1).cuda()  # [bs, seq_length, args.n_agents, 2, 2]

    rotated_matrix = torch.stack(
        [torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)], dim=1). \
        view(-1, 2, 2).cuda()

    # (5) rotate pos: including ally and enemy
    ally_pos = reshaped_ally_feats[:, :, :, :, 2:4].clone()
    enemy_pos = reshaped_enemy_feats[:, :, :, :, 2:4].clone()

    ally_p = reshaped_ally_feats[:, :, :, :, 2:4].unsqueeze(-2)
    rotated_ally_pos = (ally_p @ ally_pos_rotation_matrix).squeeze(-2)

    enemy_p = reshaped_enemy_feats[:, :, :, :, 2:4].unsqueeze(-2)
    rotated_enemy_pos = (enemy_p @ enemy_pos_rotation_matrix).squeeze(-2)

    # rotated_ally_pos = torch.matmul(ally_pos,ally_pos_rotation_matrix)
    # rotated_enemy_pos = torch.matmul(enemy_pos,enemy_pos_rotation_matrix)
    reshaped_ally_feats[:, :, :, :, 2:4] = rotated_ally_pos
    reshaped_enemy_feats[:, :, :, :, 2:4] = rotated_enemy_pos
    new_obs = torch.cat([move_feats,
                         reshaped_enemy_feats.view(bs, seq_length, args.n_agents, -1),
                         reshaped_ally_feats.view(bs, seq_length, args.n_agents, -1),
                         own_feats], dim=-1)

    # for debug
    # visualize_pos(ally_pos[0, 0, 3].squeeze().cpu().numpy(), rotated_ally_pos[0,0,3].squeeze().cpu().numpy(), theta[0].item())

    # (5) transform move actions and available actions responding with rotation angle theta
    direct_vector = torch.tensor([[0, 1],
                                  [0, -1],
                                  [1, 0],
                                  [-1, 0]]).view(1, 1, 1, 4, 2).repeat(bs, seq_length, args.n_agents, 1, 1).cuda()

    # move action
    move_vector = (direct_vector * other_action_onehot[:, :, :, 2:].unsqueeze(-1)).sum(-2)
    rotated_move_vector = (move_vector.unsqueeze(-2) @ action_rotation_matrix).squeeze(-2)
    rotated_move_action = (rotated_move_vector.unsqueeze(-2).repeat(1, 1, 1, 4, 1) * direct_vector).sum(-1).argmax(
        -1)
    rotated_move_action_onehot = torch.eye(4).cuda()[rotated_move_action]
    no_move_agent = other_action_onehot[:, :, :, :2].sum(-1)
    take_attack_agent = attack_action_onehot.sum(-1)
    rotated_move_action_onehot[no_move_agent == 1] = 0
    rotated_move_action_onehot[take_attack_agent == 1] = 0
    new_action_onehot = actions_onehot.clone()
    new_action_onehot[:, :, :, 2:attack_action_start_idx] = rotated_move_action_onehot
    new_action = new_action_onehot.argmax(-1)

    # available action
    available_move_vector = (direct_vector * avail_other_action[:, :, :, 2:].unsqueeze(-1)).float()
    # [bs, seq_length, args.n_agents, 4, 2]

    rotated_available_move_vector = available_move_vector @ action_rotation_matrix  # [bs, seq_length, args.n_agents, 4, 2]
    rotated_available_move_action = (
            rotated_available_move_vector.unsqueeze(-2).repeat(1, 1, 1, 1, 4, 1) * direct_vector.
            unsqueeze(-3).repeat(1, 1, 1, 4, 1, 1)).sum(-1).argmax(-1)



    rotated_available_move_action[avail_other_action[:, :, :, 2:] == 0] = 4
    tmp = torch.zeros(bs, seq_length, args.n_agents, 5).cuda()
    new_available_move_action = tmp.scatter(-1, rotated_available_move_action, 1)[:, :, :, 0:4]
    new_available_action = avail_actions.clone()
    new_available_action[:, :, :, 2:6] = new_available_move_action

    # (6) compute inverse permutation move action of rotation matrix
    basis_vector = torch.tensor([[0, 1],
                                 [0, -1],
                                 [1, 0],
                                 [-1, 0]]).float().cuda()
    rotated_basis_vector = basis_vector.view(1, 4, 2).repeat(bs, 1, 1) @ rotated_matrix
    move_action_permutation = (rotated_basis_vector.unsqueeze(-2).repeat(1, 1, 4, 1) * basis_vector.unsqueeze(-3).
                               repeat(1, 4, 1, 1)).sum(-1).argmax(-1)
    move_action_permutation_matrix = torch.eye(4).cuda()[move_action_permutation]
    inverse_permutation_matrix = move_action_permutation_matrix.inverse()


    # output batch data
    new_batch.data.transition_data["obs"] = new_obs
    # new_batch.data.transition_data["actions"] = new_action
    # new_batch.data.transition_data["actions_onehot"] = new_action_onehot
    # new_batch.data.transition_data["avail_actions"] = new_available_action
    new_batch.data.transition_data["inverse_matrix"] = inverse_permutation_matrix
    return new_batch

def data_symmetrize(args, batch: EpisodeBatch, axis):
    # rotation_angles = th.rand(augment_times)

    bs = batch.batch_size
    seq_length = batch.max_seq_length
    obs_component_dim = _get_obs_component_dim(args=args)
    attack_action_start_idx = 6
    augmented_data = []
    # for t in range(rotate_times):
    new_batch = copy.deepcopy(batch)
    obs = new_batch.data.transition_data["obs"].cuda()  # [bs, seq_length, n_agents, obs_dim]
    # actions = new_batch.data.transition_data["actions"]  # [bs, seq_length, n_agents, 1]
    actions_onehot = new_batch.data.transition_data["actions_onehot"].cuda()  # [bs, seq_length, n_agents, action_num]
    avail_actions = new_batch.data.transition_data["avail_actions"].cuda()  # [bs, seq_length, n_agents, action_num]
    # states = new_batch.data.transition_data["avail_actions"]
    # (1) split observation according to the semantic meaning
    move_feats, enemy_feats, ally_feats, own_feats = th.split(obs, obs_component_dim, dim=-1)
    reshaped_enemy_feats = enemy_feats.contiguous().view(bs, seq_length, args.n_agents, args.n_enemies, -1)
    reshaped_ally_feats = ally_feats.contiguous().view(bs, seq_length, args.n_agents, (args.n_agents - 1), -1)

    # (2) split available action into 2 groups: 'move' and 'attack'.
    avail_other_action = avail_actions[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
    avail_attack_action = avail_actions[:, :, :, attack_action_start_idx:]  # [n_enemies]

    # (3) split actions_onehot into 2 groups: 'move' and 'attack'.
    other_action_onehot = actions_onehot[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
    attack_action_onehot = actions_onehot[:, :, :, attack_action_start_idx:]  # [n_enemies]

    # (4) generate symmetrize obs
    ally_pos = reshaped_ally_feats[:, :, :, :, 2:4].clone().unsqueeze(-2)
    enemy_pos = reshaped_enemy_feats[:, :, :, :, 2:4].clone().unsqueeze(-2)

    if axis == 0:
        symm_matrix = torch.tensor([[-1, 0],
                                    [0, 1.]]).view(-1, 2, 2).cuda()
        move_action_permutation = torch.tensor([0, 1, 3, 2])

    elif axis == 1:
        symm_matrix = torch.tensor([[1., 0],
                                    [0, -1]]).view(-1, 2, 2).cuda()
        move_action_permutation = torch.tensor([1, 0, 2, 3])


    symm_ally_pos = (ally_pos @ symm_matrix).squeeze(-2)
    symm_enemy_pos = (enemy_pos @ symm_matrix).squeeze(-2)

    reshaped_ally_feats[:, :, :, :, 2:4] = symm_ally_pos
    reshaped_enemy_feats[:, :, :, :, 2:4] = symm_enemy_pos
    new_obs = torch.cat([move_feats,
                         reshaped_enemy_feats.view(bs, seq_length, args.n_agents, -1),
                         reshaped_ally_feats.view(bs, seq_length, args.n_agents, -1),
                         own_feats], dim=-1)

    # (5) generate inverse matrix of move action permutation
    move_action_permutation_matrix = torch.eye(4)[move_action_permutation]

    inverse_permutation_matrix = move_action_permutation_matrix.inverse().cuda()


    # output batch data
    new_batch.data.transition_data["obs"] = new_obs

    new_batch.data.transition_data["inverse_matrix"] = inverse_permutation_matrix
    return new_batch

def data_random_symm_process(args, batch: EpisodeBatch, theta=None):
    if theta == None:
        theta = th.pi * th.rand(1)
    is_symm = th.randint(2,(1,)).item()

    bs = batch.batch_size
    seq_length = batch.max_seq_length
    obs_component_dim = _get_obs_component_dim(args=args)
    attack_action_start_idx = 6
    # for t in range(rotate_times):
    new_batch = copy.deepcopy(batch)
    obs = new_batch.data.transition_data["obs"].cuda()  # [bs, seq_length, n_agents, obs_dim]
    actions = new_batch.data.transition_data["actions"].cuda()  # [bs, seq_length, n_agents, 1]
    actions_onehot = new_batch.data.transition_data["actions_onehot"].cuda()  # [bs, seq_length, n_agents, action_num]
    avail_actions = new_batch.data.transition_data["avail_actions"].cuda()  # [bs, seq_length, n_agents, action_num]
    # states = new_batch.data.transition_data["avail_actions"]
    # (1) split observation according to the semantic meaning
    move_feats, enemy_feats, ally_feats, own_feats = th.split(obs, obs_component_dim, dim=-1)
    reshaped_enemy_feats = enemy_feats.contiguous().view(bs, seq_length, args.n_agents, args.n_enemies, -1)
    reshaped_ally_feats = ally_feats.contiguous().view(bs, seq_length, args.n_agents, (args.n_agents - 1), -1)

    # (2) split available action into 2 groups: 'move' and 'attack'.
    avail_other_action = avail_actions[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
    avail_attack_action = avail_actions[:, :, :, attack_action_start_idx:]  # [n_enemies]

    # (3) split actions_onehot into 2 groups: 'move' and 'attack'.
    other_action_onehot = actions_onehot[:, :, :, :attack_action_start_idx]  # (no_op, stop, up, down, right, left)
    attack_action_onehot = actions_onehot[:, :, :, attack_action_start_idx:]  # [n_enemies]

    # (4) generate orthogonal matrix
    orthogonal_matrix = torch.stack([torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)])
    if is_symm == 1:
        orthogonal_matrix[0:2] = -1 * orthogonal_matrix[0:2]    # symmetric about the Y axis

    enemy_pos_rotation_matrix = orthogonal_matrix. \
        view(1, 1, 1, 1, 2, 2).repeat(1, seq_length, args.n_agents, args.n_enemies, 1, 1).cuda()
    ally_pos_rotation_matrix = orthogonal_matrix. \
        view(1, 1, 1, 1, 2, 2).repeat(1, seq_length, args.n_agents, args.n_agents-1, 1, 1).cuda()
    # ally_pos_rotation_matrix = torch.cat(
    #         [torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)]). \
    #         view(-1, 2, 2).cuda()
    action_rotation_matrix = orthogonal_matrix. \
        view(-1, 1, 1, 2, 2).repeat(1, seq_length, args.n_agents, 1,
                                    1).cuda()  # [bs, seq_length, args.n_agents, 2, 2]

    rotated_matrix = orthogonal_matrix. \
        view(-1, 2, 2).cuda()

    # (5) rotate pos: including ally and enemy

    ally_p = reshaped_ally_feats[:, :, :, :, 2:4].unsqueeze(-2)
    rotated_ally_pos = (ally_p @ ally_pos_rotation_matrix).squeeze(-2)

    enemy_p = reshaped_enemy_feats[:, :, :, :, 2:4].unsqueeze(-2)
    rotated_enemy_pos = (enemy_p @ enemy_pos_rotation_matrix).squeeze(-2)

    # rotated_ally_pos = torch.matmul(ally_pos,ally_pos_rotation_matrix)
    # rotated_enemy_pos = torch.matmul(enemy_pos,enemy_pos_rotation_matrix)
    reshaped_ally_feats[:, :, :, :, 2:4] = rotated_ally_pos
    reshaped_enemy_feats[:, :, :, :, 2:4] = rotated_enemy_pos
    new_obs = torch.cat([move_feats,
                         reshaped_enemy_feats.view(bs, seq_length, args.n_agents, -1),
                         reshaped_ally_feats.view(bs, seq_length, args.n_agents, -1),
                         own_feats], dim=-1)

    # for debug
    # visualize_pos(ally_pos[0, 0, 3].squeeze().cpu().numpy(), rotated_ally_pos[0,0,3].squeeze().cpu().numpy(), theta[0].item())

    # (5) transform move actions and available actions responding with rotation angle theta
    direct_vector = torch.tensor([[0, 1],
                                  [0, -1],
                                  [1, 0],
                                  [-1, 0]]).view(1, 1, 1, 4, 2).repeat(bs, seq_length, args.n_agents, 1, 1).cuda()

    # move action
    move_vector = (direct_vector * other_action_onehot[:, :, :, 2:].unsqueeze(-1)).sum(-2)
    rotated_move_vector = (move_vector.unsqueeze(-2) @ action_rotation_matrix).squeeze(-2)
    rotated_move_action = (rotated_move_vector.unsqueeze(-2).repeat(1, 1, 1, 4, 1) * direct_vector).sum(-1).argmax(
        -1)
    rotated_move_action_onehot = torch.eye(4).cuda()[rotated_move_action]
    no_move_agent = other_action_onehot[:, :, :, :2].sum(-1)
    take_attack_agent = attack_action_onehot.sum(-1)
    rotated_move_action_onehot[no_move_agent == 1] = 0
    rotated_move_action_onehot[take_attack_agent == 1] = 0
    new_action_onehot = actions_onehot.clone()
    new_action_onehot[:, :, :, 2:attack_action_start_idx] = rotated_move_action_onehot
    new_action = new_action_onehot.argmax(-1).unsqueeze(-1)

    # available action
    available_move_vector = (direct_vector * avail_other_action[:, :, :, 2:].unsqueeze(-1)).float()
    # [bs, seq_length, args.n_agents, 4, 2]

    rotated_available_move_vector = available_move_vector @ action_rotation_matrix  # [bs, seq_length, args.n_agents, 4, 2]
    rotated_available_move_action = (
            rotated_available_move_vector.unsqueeze(-2).repeat(1, 1, 1, 1, 4, 1) * direct_vector.
            unsqueeze(-3).repeat(1, 1, 1, 4, 1, 1)).sum(-1).argmax(-1)


    rotated_available_move_action[avail_other_action[:, :, :, 2:] == 0] = 4  #
    tmp = torch.zeros(bs, seq_length, args.n_agents, 5).cuda()
    new_available_move_action = tmp.scatter(-1, rotated_available_move_action, 1)[:, :, :, 0:4]
    new_available_action = avail_actions.clone()
    new_available_action[:, :, :, 2:6] = new_available_move_action

    # (6) compute inverse permutation move action of rotation matrix
    basis_vector = torch.tensor([[0, 1],
                                 [0, -1],
                                 [1, 0],
                                 [-1, 0]]).float().cuda()
    rotated_basis_vector = basis_vector.view(1, 4, 2).repeat(bs, 1, 1) @ rotated_matrix
    move_action_permutation = (rotated_basis_vector.unsqueeze(-2).repeat(1, 1, 4, 1) * basis_vector.unsqueeze(-3).
                               repeat(1, 4, 1, 1)).sum(-1).argmax(-1)
    move_action_permutation_matrix = torch.eye(4).cuda()[move_action_permutation]
    inverse_permutation_matrix = move_action_permutation_matrix.inverse()

    # ori_move_action = (rotated_move_action_onehot.unsqueeze(-2) @ inverse_permutation_matrix.view(64, 1, 1, 4, 4)).squeeze(-2)

    #  state
    state_component = args.state_component
    state_nf_al = args.state_ally_feats_size
    state_nf_en = args.state_enemy_feats_size
    state = batch.data.transition_data["state"].cuda()

    ally_feats = state[:, :, :state_component[0]].reshape(bs, seq_length, args.n_agents, state_nf_al)
    enemy_feats = state[:, :, state_component[0]: state_component[0] + state_component[1]]. \
        reshape(bs, seq_length, args.n_enemies, state_nf_en)


    ally_pos = ally_feats[:, :, :, 2:4].unsqueeze(-2)
    enemy_pos = enemy_feats[:, :, :, 1:3].unsqueeze(-2)

    state_rotate_matrix = orthogonal_matrix. \
        view(1, 1, 1, 2, 2).repeat(1, 1, args.n_agents, 1, 1).cuda()

    transferred_ally_pos = ally_pos @ state_rotate_matrix
    transferred_enemy_pos = enemy_pos @ state_rotate_matrix

    ally_feats[:, :, :, 2:4] = transferred_ally_pos.squeeze(-2)
    enemy_feats[:, :, :, 1:3] = transferred_enemy_pos.squeeze(-2)

    new_state = state.clone()
    new_state[:,:, :state_component[0]] = ally_feats.reshape(bs, seq_length, state_component[0])
    new_state[:,:, state_component[0]: state_component[0] + state_component[1]] = enemy_feats.reshape(bs, seq_length, state_component[1])


    # output batch data
    new_batch.data.transition_data["obs"] = new_obs
    new_batch.data.transition_data["state"] = new_state
    new_batch.data.transition_data["actions"] = new_action
    new_batch.data.transition_data["actions_onehot"] = new_action_onehot
    new_batch.data.transition_data["avail_actions"] = new_available_action
    new_batch.data.transition_data["inverse_matrix"] = inverse_permutation_matrix
    return new_batch




if __name__ == "__main__":
    class args:
        def __init__(self, n_agents, n_enenmies):
            self.n_agents = n_agents
            self.n_enemies = n_enenmies
            self.n_allies = n_agents - 1
            self.obs_component = [4, (10, 8), (9, 8), 6]
            # protoss: [4, (10, 9), (9, 9), 7], terran: [4, (10, 8), (9, 8), 6], zerg:[4, (10, 8), (9, 8), 6]
            self.state_component = [70, 60, 160]
            # protoss:[80, 70, 160], terran:[70, 60, 160], zerg:[70, 60, 160]
            self.state_ally_feats_size = 7  # protoss 8, terran 7, zerg 7
            self.state_enemy_feats_size = 6  # protoss 7, terran 6, zerg 6


    args = args(10, 10)

    batchs = torch.load("data_for_evaluate/10gen_zerg_batchs.th")
    # new_batch = data_random_rotate(args, batch)


    new_batchs = []
    for batch in batchs:
        new_batch = data_random_symm_process(args, batch)
        new_batchs.append(new_batch)
    torch.save(new_batchs + batchs, "data_for_evaluate/zerg_batchs.th")
    print("ok")
