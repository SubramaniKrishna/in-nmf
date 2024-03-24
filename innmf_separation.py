# iN-NMF separation
"""
Pipeline:
1. Obtain input audio
2. Train iN-NMF for single N
3. Mix and learn H for mixture by sampling W at appropriate frequencies
4. Compute separation metrics
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import numpy as np
from utils import timit_train_sources_mix
import tqdm
import json
from models import torch_stft
from datetime import datetime
import glob
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import innmf_W, binary_sep

now = datetime.now()
from pystoi import stoi
import mir_eval

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
N = 100
p = 0.5
algo = 'mse'
list_metrics = ['sdr','sir','sar', 'stoi']
rank = 20
list_N = np.array([1000,1500,2000,2500])
sigma2 = 0.0

model_results = {(str)(k):{j:[] for j in list_metrics} for k in list_N}
model_results["power"] = p
model_results["Number_runs"] = N
model_results["Algorithm"] = algo
model_results["rank"] = rank
model_results["list_N"] = list_N.tolist()
model_results["sigma2"] = sigma2

list_nmf_metrics_N = []
eps = 1.0e-6

list_train_N = [2000]

model_results["list_train_N"] = list_train_N

for e in tqdm.trange(N):
    # Obtain random train source
    s1,s2,smix = timit_train_sources_mix(timit_root = '/mnt/data/Speech/timit/timit/')
    s1gpu = torch.from_numpy(s1).float().unsqueeze(0)
    s2gpu = torch.from_numpy(s2).float().unsqueeze(0)
    smixgpu = torch.from_numpy(smix[0]).float().unsqueeze(0)

    dict_tstft = {(str)(n):torch_stft(N = n,H = n//2) for n in list_train_N}
    dict_x1_n = {(str)(n):dict_tstft[(str)(n)](s1gpu,True,'forward').squeeze(0) for n in list_train_N}
    dict_x2_n = {(str)(n):dict_tstft[(str)(n)](s2gpu,True,'forward').squeeze(0) for n in list_train_N}
    dict_finds = {(str)(n): (torch.arange((n//2 + 1))/(n//2 + 1) - 0.5) for n in list_train_N}
    for n in list_train_N:
        Nh =  min(dict_x1_n[(str)(n)].shape[-1],dict_x2_n[(str)(n)].shape[-1])
        dict_x1_n[(str)(n)] = dict_x1_n[(str)(n)][:,:Nh]
        dict_x2_n[(str)(n)] = dict_x2_n[(str)(n)][:,:Nh]
        

    # Init models
    nmf_net_1 = innmf_W(N = None, rank = rank,Npos = 10,hidden_size = 64,trainable=False,Nh = Nh).cuda()
    nmf_net_2 = innmf_W(N = None, rank = rank,Npos = 10,hidden_size = 64,trainable=False,Nh = Nh).cuda()    
    learning_rate = 1.0e-3
    optim_WH_1 = torch.optim.Rprop(nmf_net_1.parameters(), lr=learning_rate)
    optim_WH_2 = torch.optim.Rprop(nmf_net_2.parameters(), lr=learning_rate)

    torch.cuda.empty_cache()
    @cudagraph
    def train( spec_x_1, spec_x_2, finds):
        optim_WH_1.zero_grad(True)
        xhat_1 = nmf_net_1(finds)
        loss1 = torch.mean(spec_x_1*(torch.log((spec_x_1 + 1.0e-8)/(xhat_1 + 1.0e-8))) - spec_x_1 + xhat_1)
        loss1.backward()
        optim_WH_1.step()

        optim_WH_2.zero_grad(True)
        xhat_2= nmf_net_2(finds)
        loss2 = torch.mean(spec_x_2*(torch.log((spec_x_2 + 1.0e-8)/(xhat_2 + 1.0e-8))) - spec_x_2 + xhat_2)
        loss2.backward()
        optim_WH_2.step()

        return loss1

    # Train
    num_epochs = 1000
    nmf_net_1.train()
    nmf_net_2.train()
    for i in tqdm.trange(num_epochs):
        n = np.random.choice(list_train_N)

        spec_x_1 = (abs(dict_x1_n[(str)(n)])**p).cuda()
        spec_x_2 = (abs(dict_x2_n[(str)(n)])**p).cuda()
        finds = dict_finds[(str)(n)].cuda()
        train(spec_x_1,spec_x_2, finds)

    # Evaluate on different sizes (also need to train before evaluating on each size)
    nmf_net_1.eval()
    nmf_net_2.eval()
    for n in tqdm.tqdm(list_N):

        tstft = torch_stft(N = n,H = n//2)
        xmix = tstft(smixgpu,True,'forward').cuda()
        spec_x_mix = (abs(xmix)**p).squeeze()
        

        Nb2 = n//2 + 1
        finds = (torch.arange(Nb2)/Nb2 - 0.5).cuda()

        nae_sep = binary_sep(rank,spec_x_mix.shape[-1],rank,spec_x_mix.shape[-1]).cuda()
        opt_sep = torch.optim.Rprop(nae_sep.parameters(), lr=learning_rate)

        W1 = nmf_net_1.get_W(finds).detach()
        W2 = nmf_net_2.get_W(finds).detach()

        torch.cuda.empty_cache()
        @cudagraph
        def train(spec_x_mix, W1,W2):
            opt_sep.zero_grad(True)
            x_hat_mix = nae_sep(W1,W2,True)
            loss = torch.mean(spec_x_mix*(torch.log((spec_x_mix + 1.0e-8)/(x_hat_mix + 1.0e-8))) - spec_x_mix + x_hat_mix)
            loss.backward()
            opt_sep.step()

            return loss

        num_epochs = 100
        nae_sep.train()
        for epoch in tqdm.trange(num_epochs):
            train(spec_x_mix,W1,W2)
        
        nae_sep.eval()
        estimate_1 = (torch.matmul(nmf_net_1.get_W(finds),nae_sep.splus(nae_sep.H1)))
        estimate_2 = (torch.matmul(nmf_net_2.get_W(finds),nae_sep.splus(nae_sep.H2)))
        estimate_sum = estimate_1 + estimate_2
        mask1 = estimate_1/estimate_sum
        mask2 = estimate_2/estimate_sum

        stft_recon_1 = (((mask1*spec_x_mix)**(1.0/p))*torch.exp(1.0j*torch.angle(xmix))).cpu()
        stft_recon_2 = (((mask2*spec_x_mix)**(1.0/p))*torch.exp(1.0j*torch.angle(xmix))).cpu()

        component_1 = tstft(stft_recon_1.cfloat(),True,'inverse').squeeze().detach()
        component_2 = tstft(stft_recon_2.cfloat(),True,'inverse').squeeze().detach()

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

results_file = dir_pth_save + str(now) + '_innmf_separation.json'

with open(results_file, 'w') as fp:
    json.dump(model_results, fp)












