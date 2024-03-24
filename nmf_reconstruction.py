# Classical NMF reconstructions
"""
Pipeline:
1. Obtain input audio
2. Perform Classical NMF (for different N)
3. Measure reconstruction performance and dump
"""

import numpy as np
from utils import timit_train_sources_mix
from nmf_models import *
import tqdm
import json
from models import torch_stft
from datetime import datetime
import glob
import torch

now = datetime.now()

# Number of experiments
N = 100
p = 0.5
algo = 'kld'
list_metrics = ['msen_freq','sdr_freq','kld_freq','is_freq','msen_time','sdr_time']
rank = 20
list_N = np.array([1000,1500,2000,2500])

model_results = {(str)(k):{j:[] for j in list_metrics} for k in list_N}
model_results["power"] = p
model_results["Number_runs"] = N
model_results["Algorithm"] = algo
model_results["rank"] = rank
model_results["list_N"] = list_N.tolist()

list_nmf_metrics_N = []
eps = 1.0e-6

for n in tqdm.tqdm(list_N):
    for e in tqdm.trange(N):
        tstft = torch_stft(N = n,H = n//2)
        s,_,_ = timit_train_sources_mix(timit_root = '/mnt/data/Speech/timit/timit/')

        x = tstft(torch.from_numpy(s).float().unsqueeze(0),True,'forward')
        spec_x = (abs(x)**p).squeeze().numpy()

        W,H = nmf(spec_x,rank,algo = algo)
        xhat = W@H

        # Compute metrics
        model_results[str(n)]['msen_freq'].append(np.mean(np.abs(xhat**(1.0/p) - spec_x**(1.0/p))**2)/np.mean(np.abs(spec_x**(1.0/p) + eps)**2))
        model_results[str(n)]['sdr_freq'].append(20*np.log10(np.mean((np.abs(spec_x**(1.0/p)))/np.mean((np.abs(xhat**(1.0/p) - spec_x**(1.0/p) + eps))))))
        model_results[str(n)]['kld_freq'].append(np.mean(spec_x*np.log(spec_x/(xhat + eps)) - (spec_x - xhat)))
        model_results[str(n)]['is_freq'].append(np.mean((spec_x/(xhat + eps)) - np.log(spec_x/(xhat + eps)) + 1))

        recon_x_time = tstft((abs(torch.from_numpy(xhat).float())**(1.0/p))*torch.exp(1.0j*torch.angle(x)),True,'inverse').squeeze().numpy()

dir_pth_save = './'

results_file = dir_pth_save + str(now) + '_' + algo + '_classic_reconstruction.json'

with open(results_file, 'w') as fp:
    json.dump(model_results, fp)










