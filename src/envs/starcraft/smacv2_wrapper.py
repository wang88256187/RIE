from smacv2.env.starcraft2.distributions import get_distribution
from smacv2.env.starcraft2.original_sc2 import StarCraft2Env
from smacv2.env import MultiAgentEnv

map_config = {"10gen_terran": "/home/wangdongzi/my_expriments/smacv2/smacv2/examples/configs/sc2_gen_terran.yaml",
              "10gen_protoss":  "/home/wangdongzi/my_expriments/smacv2/smacv2/examples/configs/sc2_gen_protoss.yaml",
              "10gen_zerg":  "/home/wangdongzi/my_expriments/smacv2/smacv2/examples/configs/sc2_gen_zerg.yaml"
              }


class StarCraftCapabilityEnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        self.env = StarCraft2Env(**kwargs)
        self._init()
        assert (
            self.distribution_config.keys()
            == kwargs["capability_config"].keys()
        ), "Must give distribution config and capability config the same keys"

    def _init(self):
        self.n_agents = self.distribution_config["n_units"]
        self.n_enemies = self.distribution_config["n_enemies"]
        self.unit_type_bits = self.env.unit_type_bits
        self.shield_bits_ally = self.env.shield_bits_ally



    def _parse_distribution_config(self):
        for env_key, config in self.distribution_config.items():
            if env_key == "n_units" or env_key == "n_enemies":
                continue
            config["env_key"] = env_key
            # add n_units key
            config["n_units"] = self.distribution_config["n_units"]
            config["n_enemies"] = self.distribution_config["n_enemies"]
            distribution = get_distribution(config["dist_type"])(config)
            self.env_key_to_distribution_map[env_key] = distribution

    def reset(self):
        reset_config = {}
        for distribution in self.env_key_to_distribution_map.values():
            reset_config = {**reset_config, **distribution.generate()}

        return self.env.reset(reset_config)

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError


    def get_obs_component(self):
        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()
        obs_component = [move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim]
        return obs_component


    def get_state_component(self):
        if self.obs_instead_of_state:
            return [self.get_obs_size()] * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = [ally_state, enemy_state]

        if self.state_last_action:
            size.append(self.n_agents * self.n_actions)
        if self.state_timestep_number:
            size.append(1)
        return size

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "n_enemies": self.n_enemies,
            "episode_limit": self.env.episode_limit,
            "obs_ally_feats_size": self.get_obs_ally_feats_size(),
            "obs_enemy_feats_size": self.get_obs_enemy_feats_size(),
            "state_ally_feats_size": 4 + self.shield_bits_ally + self.unit_type_bits,
            "state_enemy_feats_size": 3 + self.shield_bits_enemy + self.unit_type_bits,
            "obs_component": self.get_obs_component(),
            "state_component": self.get_state_component(),
            "map_type": self.map_type,
        }
        print(env_info)
        return env_info



    def get_obs(self):
        return self.env.get_obs()

    def get_obs_move_feats_size(self):
        return self.env.get_obs_move_feats_size()

    def get_obs_own_feats_size(self):
        return self.env.get_obs_own_feats_size()

    def get_obs_ally_feats_size(self):
        return self.env.get_obs_ally_feats_size()

    def get_obs_enemy_feats_size(self):
        return self.env.get_obs_enemy_feats_size()

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        return self.env.get_state()

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()


    def get_avail_actions(self):
        return self.env.get_avail_actions()

    # def get_env_info(self):
    #     return self.env.get_env_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self):
        return self.env.render()

    def step(self, actions):
        return self.env.step(actions)

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()

    def _get_medivac_ids(self):
        medivac_ids = []
        for al_id, al_unit in self.agents.items():
            if self.map_type in ["MMM", "terran_gen"] and al_unit.unit_type == self.env.medivac_id:
                medivac_ids.append(al_id)
        print(medivac_ids)  # [9]
        return medivac_ids
