# Classical NMF separation
"""
Pipeline:
1. Obtain input audio
2. Perform Classical NMF (for different N) and obtain dictionaries
3. Mix and learn H for mixture
4. Compute separation metrics
"""

import numpy as np
# np.random.seed(10)
from utils import timit_train_sources_mix
from nmf_models import *
import tqdm
import json
from models import torch_stft
from datetime import datetime
import glob
import torch
# import fast_bss_eval
import mir_eval
from pystoi import stoi

now = datetime.now()

# Number of experiments
N = 100
p = 0.5
algo = 'kld'
list_metrics = ['sdr','sir','sar', 'stoi']
rank = 20
list_N = np.array([1000,1500,2000,2500])

model_results = {(str)(k):{j:[] for j in list_metrics} for k in list_N}
model_results["power"] = p
model_results["Number_runs"] = N
model_results["Algorithm"] = algo
model_results["rank"] = rank
# model_results["list_N"] = list_N

list_nmf_metrics_N = []
eps = 1.0e-6

for n in tqdm.tqdm(list_N):
    for e in tqdm.trange(N):
        tstft = torch_stft(N = n,H = n//2)
        s1,s2,smix = timit_train_sources_mix(timit_root = '/mnt/data/Speech/timit/timit/')

        x1 = tstft(torch.from_numpy(s1).float().unsqueeze(0),True,'forward')
        x2 = tstft(torch.from_numpy(s2).float().unsqueeze(0),True,'forward')
        xmix = tstft(torch.from_numpy(smix[0]).float().unsqueeze(0),True,'forward')
        spec_s1 = (abs(x1)**p).squeeze().numpy()
        spec_s2 = (abs(x2)**p).squeeze().numpy()
        spec_smix = (abs(xmix)**p).squeeze().numpy()

        # W,H = nmf(spec_x,20,algo = algo)
        W1,H1 = nmf(spec_s1,rank,algo = algo)
        W2,H2 = nmf(spec_s2,rank,algo = algo)

        estimate_1,estimate_2 = nmf_h1h2(spec_smix,W1,W2,rank)
        estimate_sum = estimate_1 + estimate_2
        mask1 = estimate_1/estimate_sum
        mask2 = estimate_2/estimate_sum

        stft_recon_1 = ((mask1*spec_smix)**(1.0/p))*np.exp(1.0j*np.angle(xmix))
        stft_recon_2 = ((mask2*spec_smix)**(1.0/p))*np.exp(1.0j*np.angle(xmix))

        component_1 = tstft(torch.from_numpy(stft_recon_1).cfloat(),True,'inverse').squeeze().detach().cpu()
        component_2 = tstft(torch.from_numpy(stft_recon_2).cfloat(),True,'inverse').squeeze().detach().cpu()

        ground_truth = np.vstack([smix[1],smix[2]])
        separated_components = np.vstack([component_1,component_2])
        indmin = min(ground_truth.shape[1],separated_components.shape[1])
        (sdr, sir, sar,
        _) = mir_eval.separation.bss_eval_sources(ground_truth[:,:indmin],separated_components[:,:indmin])
        stoi_1 = stoi(ground_truth[0,:indmin],separated_components[0,:indmin],16000)
        stoi_2 = stoi(ground_truth[1,:indmin],separated_components[1,:indmin],16000)

        model_results[str(n)]['sdr'].append(sdr.mean())
        model_results[str(n)]['sir'].append(sir.mean())
        model_results[str(n)]['sar'].append(sar.mean())
        model_results[str(n)]['stoi'].append(0.5*(stoi_1 + stoi_2))


dir_pth_save = './'

results_file = dir_pth_save + str(now) + '_' + algo + '_classic_separation.json'

with open(results_file, 'w') as fp:
    json.dump(model_results, fp)










