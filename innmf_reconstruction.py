# iN-NMF Reconstructions
"""
Pipeline:
1. Obtain input audio
2. Train iN-NMF models for one N, and sample W at different N's frequencies to update H only
3. Measure reconstruction performance and dump
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
from utils import timit_train_sources_mix
import tqdm
import json
from datetime import datetime
import glob
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import innmf_W,torch_stft

now = datetime.now()

import gc

# Decorator to optimize CUDA calls
def cudagraph( f):
    _graphs = {}
    def f_( *args):
        key = hash( tuple( tuple( a.shape) for a in args))
        if key in _graphs:
            wrapped, *_ = _graphs[key]
            return wrapped( *args)

        g = torch.cuda.CUDAGraph()
        in_tensors = [a.clone() for a in args]
        f( *in_tensors) # stream warmup

        with torch.cuda.graph( g):
            out_tensors = f( *in_tensors)

        def wrapped( *args):
            [a.copy_(b) for a, b in zip( in_tensors, args)]
            g.replay()
            return out_tensors

        _graphs[key] = (wrapped, g, in_tensors, out_tensors)
        return wrapped( *args)

    return f_

# Number of experiments
N = 10
p = 0.5
algo = 'kld'
nmodels = 10
list_metrics = ['msen_freq','sdr_freq','kld_freq','is_freq','msen_time','sdr_time']
rank = 20
nmodels = 5
list_N = np.array([1000,1500,2000,2500])
sigma2 = 0.0

model_results = {(str)(k):{j:[] for j in list_metrics} for k in list_N}
model_results["power"] = p
model_results["Number_runs"] = N
model_results["Algorithm"] = algo
model_results["rank"] = rank
model_results["list_N"] = list_N.tolist()
model_results["sigma2"] = sigma2
model_results["nmodels"] = nmodels

list_nmf_metrics_N = []
eps = 1.0e-6

list_train_N = [2000]

model_results["list_train_N"] = list_train_N

for e in tqdm.trange(N):
    # Obtain random train source
    s,_,_ = timit_train_sources_mix(timit_root = '/mnt/data/Speech/timit/timit/')
    sgpu = torch.from_numpy(s).float().unsqueeze(0)

    dict_tstft = {(str)(n):torch_stft(N = n,H = n//2) for n in list_train_N}
    dict_x_n = {(str)(n):dict_tstft[(str)(n)](sgpu,True,'forward').squeeze(0)for n in list_train_N}
    dict_finds = {(str)(n): (torch.arange((n//2 + 1))/(n//2 + 1) - 0.5) for n in list_train_N}

    for n in list_train_N:
        Nh = dict_x_n[(str)(n)].shape[-1]

    # Init models
    nmf_net = innmf_W(N = None, rank = rank,Npos = 10,hidden_size = 64,trainable=False,Nh = Nh).cuda()
    learning_rate = 1.0e-3
    optim_WH = torch.optim.Rprop(nmf_net.parameters(), lr=learning_rate)

    torch.cuda.empty_cache()
    @cudagraph
    def train( spec_x, finds):
        optim_WH.zero_grad(set_to_none=True)
        xhat = nmf_net(finds)
        loss = torch.mean(spec_x*(torch.log((spec_x + 1.0e-8)/(xhat + 1.0e-8))) - spec_x + xhat)
        loss.backward()
        optim_WH.step()
        return loss

    # Train
    num_epochs = 1000
    nmf_net.train()
    for i in tqdm.trange(num_epochs):
        n = np.random.choice(list_train_N)
        spec_x = (abs(dict_x_n[(str)(n)])**p).cuda()
        finds = dict_finds[(str)(n)].cuda()
        train(spec_x, finds)
        del spec_x,finds
    
    # Evaluate on different sizes (only train H and keep W fixed after sampling)
    nmf_net.eval()
    for n in tqdm.tqdm(list_N):

        tstft = torch_stft(N = n,H = n//2).to(device)
        x = tstft(sgpu.to(device),True,'forward')
        x = x[:,:,:nmodels*(x.shape[-1]//nmodels)]
        spec_x = (abs(x)**p)

        Nb2 = n//2 + 1
        Nt2 = spec_x.shape[2]
        finds = (torch.arange(Nb2)/Nb2 - 0.5)
        tinds = (torch.arange(Nt2//nmodels)/(Nt2/nmodels) - 0.5)
        Wnew = nmf_net.get_W(finds).detach()

        H = torch.nn.Parameter(torch.rand(rank,spec_x.shape[-1]).cuda(),requires_grad=True)
        optim_new = torch.optim.Rprop([H], lr=learning_rate)

        @cudagraph
        def train_new(spec_x):
            optim_new.zero_grad(set_to_none=True)
            xhat = Wnew@torch.nn.Softplus()(H)
            loss_H = torch.mean(spec_x*(torch.log((spec_x + 1.0e-8)/(xhat + 1.0e-8))) - spec_x + xhat)
            loss_H.backward()
            optim_new.step()
        
        num_epochs = 1000
        larr_H = []
        for i in tqdm.trange(num_epochs):
            loss = train_new(spec_x)

        
        xhat = (Wnew@torch.nn.Softplus()(H)).detach().cpu().numpy()
        spec_x = spec_x.detach().squeeze().cpu().numpy()

        # Compute metrics
        model_results[str(n)]['msen_freq'].append((np.mean(np.abs(xhat**(1.0/p) - spec_x**(1.0/p))**2)/np.mean(np.abs(spec_x**(1.0/p) + eps)**2)).astype(float))
        model_results[str(n)]['sdr_freq'].append((20*np.log10(np.mean((np.abs(spec_x**(1.0/p)))/np.mean((np.abs(xhat**(1.0/p) - spec_x**(1.0/p) + eps)))))).astype(float))
        model_results[str(n)]['kld_freq'].append((np.mean(spec_x*np.log(spec_x/(xhat + eps)) - (spec_x - xhat))).astype(float))
        model_results[str(n)]['is_freq'].append((np.mean((spec_x/(xhat + eps)) - np.log(spec_x/(xhat + eps)) + 1)).astype(float))


dir_pth_save = './'

results_file = dir_pth_save + str(now) + '_' + algo + '_innmf_reconstruction.json'

with open(results_file, 'w') as fp:
    json.dump(model_results, fp)
