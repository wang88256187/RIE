#!/bin/sh
env="smacv2"
map="10gen_terran"
algo="qmix"
graph_state=False

seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python ../../../main.py --config=${algo} --env-config="sc2_gen_protoss" with env_args.map_name=${map} obs_agent_id=True obs_last_action=False runner=parallel batch_size_run=4 buffer_size=2500 t_max=10050000 epsilon_anneal_time=100000 batch_size=64 td_lambda=0.6 use_graph_state=${graph_state}
done