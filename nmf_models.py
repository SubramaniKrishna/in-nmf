# Classical NMF Algorithm with Multiplicative Updates

import numpy as np

# Perform nmf with ('mse', 'kld', or 'is')
def nmf( x, k, algo = 'kld',iters=3000):
    w = 10 + np.random.randn( x.shape[0], k)
    h = 10 + np.random.randn( k, x.shape[1])
    eps = .000001 # use this to avoid potential divisions by 0
    for i in range( iters):
        if algo == 'mse':
            h = h * w.T.dot( x) / (w.T.dot( w).dot( h)+eps)
            w = w * x.dot( h.T) / (w.dot( h).dot( h.T)+eps)
        elif algo == 'kld':
            h = h * w.T.dot( x/(w.dot(h) + eps)) / (w.T.dot(np.ones_like(x)) + eps)
            w = w * (x/(w.dot(h) + eps)).dot( h.T) / ((np.ones_like(x)).dot( h.T) + eps)
        else:
            h = h * w.T.dot( x/(w.dot(h) + eps)**2) / (w.T.dot(np.ones_like(x)/(w.dot(h) + eps)) + eps)
            w = w * (x/(w.dot(h) + eps)**2).dot( h.T) / ((np.ones_like(x)/(w.dot(h) + eps)).dot( h.T) + eps)

    return w,h

def nmf_h1h2(x,w1,w2,k,algo = 'kld',iters=3000):
    w = np.concatenate([w1,w2],axis = -1)
    h = 10 + np.random.randn( 2*k, x.shape[1])
    eps = .000001 # use this to avoid potential divisions by 0
    for i in range( iters):
        if algo == 'mse':
            h = h * w.T.dot( x) / (w.T.dot( w).dot( h)+eps)
        elif algo == 'kld':
            h = h * w.T.dot( x/(w.dot(h) + eps)) / (w.T.dot(np.ones_like(x)) + eps)
        else:
            h = h * w.T.dot( x/(w.dot(h) + eps)**2) / (w.T.dot(np.ones_like(x)/(w.dot(h) + eps)) + eps)
    h1 = h[:k,:]
    h2 = h[k:2*k,:]
    return w1@h1,w2@h2
