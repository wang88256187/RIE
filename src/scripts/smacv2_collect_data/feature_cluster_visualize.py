import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import copy
from sklearn.manifold import TSNE

from modules.mixers.nmix import Mixer
# rie, hpn, qmix
model_path = ["/home/wangdongzi/Desktop/20230620/wangdongzi/IRMARL_0620/src/results/models/sc2wrapped_10gen_zerg-nagents_10_nenemies_10_obs_aid=1-obs_act=0/algo=ir_qmix-agent=hpns_rnn/env_n=4/mixer=qmix-hpn_hyperdim=64-acti=relu/rnn_dim=64-2bs=2500_64-tdlambda=0.6-epdec_0.05=100k/ir_qmix_seed_2__2023-06-20_08-07-12/10050022/mixer.th",
              "/home/wangdongzi/Desktop/20230620/wangdongzi/IRMARL_0620/src/results/models/sc2wrapped_10gen_zerg-nagents_10_nenemies_10_obs_aid=1-obs_act=0/algo=hpn_qmix-agent=hpns_rnn/env_n=4/mixer=qmix-hpn_hyperdim=64-acti=relu/rnn_dim=64-2bs=2500_64-tdlambda=0.6-epdec_0.05=100k/hpn_qmix_seed_3__2023-06-20_08-04-24/10050062/mixer.th",
              "/home/wangdongzi/extra2/ir_exp_data/models/zerg/qmix/mixer.th"]


hpn_args = torch.load("data_for_evaluate/zerg_args.pth")
rie_args = copy.deepcopy(hpn_args)
rie_args.use_graph_state = True

hpn_mixer = Mixer(hpn_args).cuda()
rie_mixer = Mixer(rie_args).cuda()
qmix_mixer = Mixer(hpn_args).cuda()
rie_mixer.load_state_dict(torch.load(model_path[0]))
hpn_mixer.load_state_dict(torch.load(model_path[1]))
qmix_mixer.load_state_dict(torch.load(model_path[2]))
# process data
batchs = torch.load("data_for_evaluate/zerg_batchs.th")

states = []
valid_index = []
for batch in batchs:
    states.append(batch["state"].cuda())
    valid_index.append(batch["filled"].cuda())

states = torch.stack(states).view(-1,1,290)
valid_index = torch.stack(valid_index).flatten()
data_length = states.shape[0]
print("data length: ", data_length)


hpn_ws=[]
rie_ws=[]
qmix_ws=[]
qvals = torch.zeros((2000, 1, 10)).cuda()

index = list(range(data_length)[::2000])
k = len(index)

with torch.no_grad():
    for n, i in enumerate(index):
        state = states[i:i+2000]
        if n == k-1:
            l = state.shape[0]
            qvals = torch.zeros((l, 1, 10)).cuda()
        hpn_w1, hpn_w2 = hpn_mixer(qvals, state)
        hpn_ws.append(hpn_w2)
        rie_w1, rie_w2 = rie_mixer(qvals, state)
        rie_ws.append(rie_w2)
        qmix_w1, qmix_w2 = qmix_mixer(qvals, state)
        qmix_ws.append(qmix_w2)

rie_ws = torch.cat(rie_ws, dim=0).view(-1,32)[valid_index==1][::10]
hpn_ws = torch.cat(hpn_ws, dim=0).view(-1,32)[valid_index==1][::10]
qmix_ws = torch.cat(qmix_ws, dim=0).view(-1,32)[valid_index==1][::10]

print("rie_ws.shape: ", rie_ws.shape)
print("hpn_ws.shape: ", hpn_ws.shape)
print("hpn_ws.shape: ", qmix_ws.shape)



x_value = torch.stack([rie_ws, hpn_ws, qmix_ws]).view(-1,32).cpu().numpy()
y_value_0 = torch.zeros_like(rie_ws)
y_value_1 = torch.ones_like(hpn_ws)
y_value_2 = torch.ones_like(qmix_ws) * 2
y_value = torch.stack([y_value_0, y_value_1, y_value_2]).view(-1,1).cpu().numpy()

print("start cluster ...")
embeddings = TSNE().fit_transform(x_value)  # t-SNE降维，默认降为二维
print("cluster done !")

vis_x = embeddings[:, 0]  # 0维
vis_y = embeddings[:, 1]  # 1维

colors = ['orchid', 'darkseagreen', 'cornflowerblue', 'peru']

plt.figure(figsize=(10, 10))
t = rie_ws.shape[0]
plt.scatter(vis_x[:t], vis_y[:t], c=colors[0], label='RIE')
plt.scatter(vis_x[t:2*t], vis_y[t:2*t], c=colors[1], label='HPN')
plt.scatter(vis_x[2*t:], vis_y[2*t:], c=colors[2], label='QMIX')


plt.legend(fontsize=20, loc='lower right')
plt.savefig('plot/feature_cluster_vis3.pdf', dpi=400)