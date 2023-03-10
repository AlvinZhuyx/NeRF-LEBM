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

class EBM_deep(nn.Module):
    def __init__(self, nz, ndf, nout, prior_type, act_type):
        super(EBM_deep, self).__init__()
        # 'ebm' means ebm prior 'gaussian' means gaussian prior
        assert prior_type in ['ebm', 'gaussian']
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
                nn.Linear(ndf, ndf),
                act,
                nn.Linear(ndf, ndf),
                act,
                nn.Linear(ndf, nout))
        else:
            self.ebm = Zero()
        
    def forward(self, z):
        return self.ebm(z).sum(dim=-1)
'''
class EBM_fm(nn.Module):
    # deal with the case that the latent variable is in shape of feature map
    # assume the feature map is in the shape of [8 * 8 * 128]
    def __init__(self, prior_type, act_type):
        super(EBM_fm, self).__init__()
        # 'ebm' means ebm prior 'gaussian' means gaussian prior
        assert prior_type in ['ebm', 'gaussian']
        if act_type == 'relu':
            act = nn.LeakyReLU(0.2)
        elif act_type == 'swish':
            act = nn.SiLU() 
        else:
            raise NotImplementedError

        if prior_type == 'ebm':

            # self.ebm = nn.Sequential(
                # nn.Conv2d(16, 32, 3, 1, 0),
                # act,
                # nn.Conv2d(32, 64, 3, 1, 0),
                # act,
                # nn.Conv2d(64, 1, 4, 4, 0))

            self.ebm = nn.Sequential(
                nn.Linear(1024, 64),
                act,
                nn.Linear(64, 64),
                act,
                nn.Linear(64, 1))
        else:
            self.ebm = Zero()
        
    def forward(self, z):
        #z_reshape = t.reshape(z, (-1, 8, 8, 16)).permute((0, 3, 1, 2)) # to (N, 16, 8, 8)
        #return self.ebm(z_reshape).sum(dim=(1, 2, 3))
        return self.ebm(z).sum(dim=-1)
'''
class EBM_feature(nn.Module):
    # deal with the case that the latent variable is in shape of feature map
    # assume the feature map is in the shape of [8 * 8 * 128]
    def __init__(self, zdim, nf, nout, prior_type, act_type):
        super(EBM_feature, self).__init__()
        # 'ebm' means ebm prior 'gaussian' means gaussian prior
        assert prior_type in ['ebm', 'gaussian']
        if act_type == 'relu':
            act = nn.LeakyReLU(0.2)
        elif act_type == 'swish':
            act = nn.SiLU() 
        else:
            raise NotImplementedError

        if prior_type == 'ebm':
            self.ebm = nn.Sequential(
                nn.Conv2d(zdim, nf, 4, 2, 1),
                act,
                nn.Conv2d(nf, nf, 4, 2, 1),
                act,
                nn.Conv2d(nf, nout, 2, 2, 0))
        else:
            self.ebm = Zero()
        
    def forward(self, z):
        return self.ebm(z).squeeze().sum(dim=-1)