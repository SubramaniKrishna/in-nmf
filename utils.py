"""
Utility code for t-f transforms and TIMIT dataloading
"""
import numpy as np
import glob
import scipy

def timit_train_sources_mix(timit_root = '/mnt/data/Speech/timit'):
    """
    Generate Two Person mixtures at 0 dB
    returns (source 1,source 2) and mixture 
    """

    dur_train = 20
    dur_test = 5

    l_source = glob.glob(timit_root + '/all/*.wav')

    while True:
        sr,source_1 = scipy.io.wavfile.read(np.random.choice(l_source))
        if source_1.shape[0]/sr > (dur_train + dur_test):
            break
    source_1 = source_1/(2**15 - 1)
    source_1 = source_1/np.sqrt(np.sum(source_1**2))

    while True:
        sr,source_2 = scipy.io.wavfile.read(np.random.choice(l_source))
        if source_2.shape[0]/sr > (dur_train + dur_test):
            break
    sr,source_2 = scipy.io.wavfile.read(np.random.choice(l_source))
    source_2 = source_2/(2**15 - 1)
    source_2 = source_2/np.sqrt(np.sum(source_2**2))

    return source_1[:(int)(sr*dur_train)],source_2[(int)(sr*dur_test):(int)(sr*(dur_train + dur_test))],[(source_1[(int)(sr*dur_train):(int)(sr*(dur_train + dur_test))] + source_2[:(int)(sr*(dur_test))]),source_1[(int)(sr*dur_train):(int)(sr*(dur_train + dur_test))],source_2[:(int)(sr*(dur_test))]]

def cqt(x,Q = 48,B = 48,num_octaves=7,f_min=100,H = 512,sr=16000):
    K = (int)(B*num_octaves)
    list_CQT_magnitudes = []
    list_CQT_times = []
    list_CQT_frequencies = []

    dur = x.shape[0]/sr

    lmin = []
    farr = []
    for k in range(K):
        f_k = (2**(k*1.0/B))*f_min
        N_k = (int)((sr*Q)/f_k)
        if(N_k%2!=0):
            N_k = N_k + 1
        H_k = N_k//16
        # H_k = H
        xpad = np.hstack([np.zeros(N_k//2),x,np.zeros(N_k//2)])

        lA = np.arange(N_k//2,xpad.shape[0] - N_k,H_k)
        lmin.append(lA.shape[0])
        farr.append(f_k)
    
    lmin = np.array(lmin)
    lmin = np.min(lmin)
    farr = np.array(farr)


    for k in range(K):
        f_k = (2**(k*1.0/B))*f_min
        N_k = (int)((sr*Q)/f_k)
        if(N_k%2!=0):
            N_k = N_k + 1
        H_k = N_k//8
        # H_k = H

        xpad = np.hstack([np.zeros(N_k//2),x,np.zeros(N_k//2)])

        W_k = np.hanning(N_k)
        e_k = np.exp((-1.0j*2*np.pi*Q*np.arange(N_k))/N_k)

        list_X = []
        lA = np.arange(N_k//2,xpad.shape[0]-N_k,H_k)
        # print(lA.shape)
        for i in lA:
            # if()
            # print(x[i - N_k//2:i+N_k//2].shape,W_k.shape)
            list_X.append(np.abs(np.sum(xpad[i - N_k//2:i+N_k//2]*W_k*e_k))/(np.sqrt(N_k)))
        # list_X = list_X[:lmin]
        # [np.abs(np.sum(x[i - N_k//2:i+N_k//2]*W_k*e_k))/(np.sqrt(N_k)) for i in np.arange(N_k//2,x.shape[0] - N_k - 1,H_k)]
        list_CQT_magnitudes = list_CQT_magnitudes + list_X
        list_CQT_times = list_CQT_times + (np.linspace(0,dur,len(list_X))).tolist()
        list_CQT_frequencies = list_CQT_frequencies + [f_k]*len(list_X)
        # print(len(list_X))

    return np.array(list_CQT_magnitudes),np.array(list_CQT_times),np.array(list_CQT_frequencies),lmin,farr

def stft(x,N = 1024,H = 512,sr=16000):
    list_stft_magnitudes = []
    list_stft_times = []
    list_stft_frequencies = []

    dur = x.shape[0]/sr

    farr = []

    for k in range(N//2 + 1):
        f_k = k*(sr/N)
        N_k = N
        H_k = H

        xpad = np.hstack([np.zeros(N_k//2),x,np.zeros(N_k//2)])

        W_k = np.hanning(N_k)
        e_k = np.exp((-1.0j*2*np.pi*k*np.arange(N_k))/N_k)

        list_X = []
        lA = np.arange(N_k//2,x.shape[0] - N_k,H_k)
        for i in lA:
            list_X.append(np.abs(np.sum(x[i - N_k//2:i+N_k//2]*W_k*e_k))/(np.sqrt(N_k)))
        # list_X = list_X[:lmin]
        list_stft_magnitudes = list_stft_magnitudes + list_X
        list_stft_times = list_stft_times + (np.linspace(0,dur,len(list_X))).tolist()
        list_stft_frequencies = list_stft_frequencies + [f_k]*len(list_X)

        farr.append(f_k)

    return np.array(list_stft_magnitudes),np.array(list_stft_times),np.array(list_stft_frequencies),np.array(farr)