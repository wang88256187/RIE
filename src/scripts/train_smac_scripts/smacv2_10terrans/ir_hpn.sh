#!/bin/sh
env="smacv2"
type="terran"
n_agents="10"
map=${n_agents}gen_${type}
config_name=sc2_gen_${type}

algo="hpn_qmix"
graph_state=True

seed_max=10

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../../../main.py --config=${algo} --env-config=${config_name} with env_args.map_name=${map} obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=4 buffer_size=2500 t_max=10050000 epsilon_anneal_time=100000 batch_size=64 td_lambda=0.6 use_graph_state=${graph_state}
done
