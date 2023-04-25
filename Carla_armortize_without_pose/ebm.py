import torch as t, torch.nn as nn

class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()
        self.W = nn.Parameter(t.tensor([0.0]))
        
    def forward(self, z):
        return 0.0 * z

class EBM(nn.Module):
    def __init__(self, nz, ndf, nout, prior_type, act_type):
        super(EBM, self).__init__()
        # 'ebm' means ebm prior 'gaussian' means gaussian prior
        assert prior_type in ['ebm', 'ebm_4layer', 'gaussian']
        if act_type == 'relu':
            act = nn.LeakyReLU(0.2)
        elif act_type == 'swish':
            act = nn.SiLU() 
        else:
            raise NotImplementedError

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
        else:
            self.ebm = Zero()
        
    def forward(self, z):
        return self.ebm(z).sum(dim=-1)