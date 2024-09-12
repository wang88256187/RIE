from functools import partial
import sys
import os

# from .multiagentenv import MultiAgentEnv

# from .starcraft.StarCraft2EnvWrapper import StarCraft2EnvWrapper
# from .starcraft.smacv2_wrapper import StarCraftCapabilityEnvWrapper

# def env_fn(env, **kwargs):
#     return env(**kwargs)
#
#
# REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2EnvWrapper)
# REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
