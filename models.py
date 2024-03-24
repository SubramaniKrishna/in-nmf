"""
PyTorch Models;
- STFT
- Random Fourier Features
- Implicit Neural Network
- Implicit-Neural NMF
"""


import torch


class torch_stft(torch.nn.Module):
    """
    Custom forward/inverse STFT using 1d conv with pre-loaded DFT kernels
    """
    def __init__(self,N = 512,H = 128):
        super( torch_stft, self).__init__()
        self.N = N
        self.H = H
        self.w = torch.sqrt(torch.hann_window(N,periodic=True,requires_grad=False)).cfloat()
        self.FFT_kernel = torch.nn.Parameter((((torch.fft.fft(torch.diag(self.w),norm = 'ortho')).transpose(1,0).unsqueeze(1).cfloat())),requires_grad = False)

    def forward(self,x,return_complex = True,xform = 'forward'):
        if (xform == 'forward'):
            x = torch.nn.functional.conv1d(x.unsqueeze(1) + 0.0j, self.FFT_kernel, stride=self.H, padding=0)
            if (return_complex == True):
                x = x[:,:self.N//2 + 1,:]
        else:
            if(return_complex == True):
                x = torch.cat([x,torch.conj(torch.flip(x[:,1:-1,:],(1,)))],1)
            x = torch.real(torch.nn.functional.conv_transpose1d( x, torch.conj(self.FFT_kernel),stride=self.H, padding=0))

        return x

class random_fourier_feature(torch.nn.Module):
    """
    Random Fourier Features as implemented in https://arxiv.org/pdf/2006.10739.pdf
    """
    def __init__(self,dim_out = 10,trainable = True):
        super( random_fourier_feature, self).__init__()
        self.B = torch.nn.Parameter(2.0**torch.arange(dim_out),requires_grad = trainable)
        self.bias = torch.nn.Parameter(torch.normal(0,1,(dim_out,)),requires_grad = trainable)
    
    def forward(self,x):
        arg = torch.vstack([self.B*torch.pi*x[i] + self.bias for i in range(x.shape[0])])
        return torch.hstack([torch.cos(arg),torch.sin(arg)])


class sin_activation(torch.nn.Module):
    """
    Sinusoidal Activation
    """
    def __init__(self):
        super(sin_activation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class neural_implicit_network(torch.nn.Module):
    """
    Implicit Neural Network, t -> Fourier Encoding(t) -> Implicit (Fourier Encoding (t))
    """
    def __init__(self,N = 32,Npos = 10, hidden_size = 64,trainable=True):
        super( neural_implicit_network, self).__init__()
        self.rff = random_fourier_feature(dim_out = Npos,trainable=trainable)
        # self.activation = torch.nn.ReLU()
        self.activation = sin_activation()
        self.nerf_h = torch.nn.ModuleList([
            torch.nn.Linear(2*Npos,hidden_size), # 2*Npos for one index only
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.Linear(hidden_size,1)
        ])
    
    def forward(self,tinds):
        ind_pe = self.rff(tinds)
        x = self.nerf_h[0](ind_pe)
        for i in range(1,len(self.nerf_h)):
            x = self.nerf_h[i](x)
            if (i < len(self.nerf_h) - 1):
                x = self.activation(x) #Residual Connections
                
        return x.permute(1,0)

class innmf_WH_both(torch.nn.Module):
    """
    Implicit Neural NMF models with both W,H Implicit
    """
    def __init__(self,N = 128,Npos = 10,hidden_size = 64,rank = 5,trainable = False):
        super( innmf_WH_both, self).__init__()
        self.N = N
        self.rank = rank
        self.splus = torch.nn.Softplus()
        self.relu = torch.nn.ReLU()
        self.W = torch.nn.ModuleDict({str(i):torch.nn.ModuleList() for i in range(rank)})
        self.H = torch.nn.ModuleDict({str(i):torch.nn.ModuleList() for i in range(rank)})
        for i in range(rank):
            self.W[str(i)].append(neural_implicit_network(N = N,Npos = Npos, hidden_size=hidden_size,trainable=trainable))
            self.H[str(i)].append(neural_implicit_network(N = N,Npos = Npos, hidden_size=hidden_size,trainable=trainable))
    
    def forward(self,finds,tinds):

        W = self.get_W(finds)
        H = self.get_H(tinds)
        xhat = (W.permute(1,0)*H).sum(0)

        return xhat
    
    def get_W(self,finds):
        W = []
        for i in range(self.rank):
            W.append(self.W[str(i)][0](finds))
        
        W = torch.cat(W)

        return self.splus((W.permute(1,0)))

    def get_H(self,tinds):
        H = []
        for i in range(self.rank):
                H.append(self.H[str(i)][0](tinds))
        
        H = torch.cat(H)

        return self.splus(H)


class innmf_W(torch.nn.Module):
    """
    Implicit W, learnable matrix H (for regular STFT)
    """
    def __init__(self,N = 128,Npos = 10,hidden_size = 64,rank = 5,trainable = False,Nh = 100):
        super( innmf_W, self).__init__()
        self.N = N
        self.rank = rank
        self.splus = torch.nn.Softplus()
        self.relu = torch.nn.ReLU()
        self.W = torch.nn.ModuleDict({str(i):torch.nn.ModuleList() for i in range(rank)})
        for i in range(rank):
            self.W[str(i)].append(neural_implicit_network(N = N,Npos = Npos, hidden_size=hidden_size,trainable=trainable))
        self.H = torch.nn.parameter.Parameter(torch.rand(self.rank,Nh),requires_grad=True)
    
    def forward(self,finds):
        W = self.get_W(finds)
        xhat = torch.matmul(W,self.splus(self.H))
        return xhat
    
    def get_W(self,finds):
        W = []
        for i in range(self.rank):
            W.append(self.W[str(i)][0](finds)) 
        W = torch.cat(W)
        return self.splus(W.permute(1,0))

class binary_sep(torch.nn.Module):
    """
    Learn factors H1,H2 for a binary separation problem given W1,W2
    """
    def __init__(self,rank1,dim1,rank2,dim2):
        super( binary_sep, self).__init__()
        self.rank1 = rank1
        self.rank2 = rank2
        self.splus = torch.nn.Softplus()

        self.H1 = torch.nn.Parameter(torch.rand(rank1,dim1),requires_grad=True)
        self.H2 = torch.nn.Parameter(torch.rand(rank2,dim2),requires_grad=True)
    
    def forward(self,W1,W2,non_negative=True):
        if non_negative:
            X_hat_1 = (torch.matmul(W1,self.splus(self.H1)))
            X_hat_2 = (torch.matmul(W2,self.splus(self.H2)))
        else:
            X_hat_1 = self.splus(torch.matmul(W1,self.H1))
            X_hat_2 = self.splus(torch.matmul(W2,self.H2))
        
        X = X_hat_1 + X_hat_2

        return X