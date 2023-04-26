import torch as t, torch.nn as nn
import torch.nn.functional as F

class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()
        self.W = nn.Parameter(t.tensor([0.0]))
        
    def forward(self, z):
        return 0.0 * z

class EBM(nn.Module):
    def __init__(self, nz, ndf, nout, prior_type, act_type, normalize=False):
        super(EBM, self).__init__()
        # 'ebm' means ebm prior 'gaussian' means gaussian prior
        assert prior_type in ['ebm', 'gaussian', 'ebm_4layer']
        if act_type == 'relu':
            act = nn.LeakyReLU(0.2)
        elif act_type == 'swish':
            act = nn.SiLU() 
        else:
            raise NotImplementedError
        self.normalize = normalize
        print('Current normalize is {}'.format(self.normalize))

        if prior_type == 'ebm':
            self.ebm = nn.Sequential(
                nn.Linear(nz, ndf),
                act,
                nn.Linear(ndf, ndf),
                act,
                nn.Linear(ndf, nout))
        elif prior_type == 'ebm_4layer':
            self.ebm = nn.Sequential(
                nn.Linear(nz, ndf),
                act,
                nn.Linear(ndf, ndf),
                act,
                nn.Linear(ndf, ndf),
                act,
                nn.Linear(ndf, ndf),
                act,
                nn.Linear(ndf, nout))
        elif prior_type == 'gaussian':
            self.ebm = Zero()
        else:
            raise NotImplementedError
        
    def forward(self, z):
        if self.normalize:
            z_ = 3 * F.normalize(z, dim=-1)
        else:
            z_ = z
        return self.ebm(z_).sum(dim=-1)