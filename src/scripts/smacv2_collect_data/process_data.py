import torch
import numpy as np

from utils.data_processing import data_random_symm_process


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

batch = torch.load("data_for_evaluate/10gen_zerg_batchs.th")[0]
# new_batch = data_random_rotate(args, batch)


new_batchs = [batch]
for _ in range(99):
    new_batch = data_random_symm_process(args, batch)
    new_batchs.append(new_batch)
torch.save(new_batchs, "data_for_evaluate/zerg_batchs.th")
print("ok")
