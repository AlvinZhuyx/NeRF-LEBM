import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torchvision import models
from hyperspherical_vae.distributions import VonMisesFisher

# this code contains the inference model that treat pose as a parameter and assume VonMisesFisher on it
# using the implementation of S-VAE https://github.com/nicola-decao/s-vae-pytorch


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2) 
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.get_device(), non_blocking=True)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

######################### Siren model ############################################################
class Sine(nn.Module):
    """Sine Activation Function."""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(30. * x)

def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.expand_as(x)
        phase_shift = phase_shift.expand_as(x)
        return torch.sin(freq * x + phase_shift)


class TALLSIREN(nn.Module):
    """Primary SIREN  architecture used in pi-GAN generators."""

    def __init__(self, input_dim=2, z_app_dim=100, z_shape_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_app_dim = z_app_dim
        self.z_shape_dim = z_shape_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        
        self.transform_network = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
          
        self.final_layer = nn.Linear(hidden_dim, 1)
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.mapping_network_shape = CustomMappingNetwork(z_shape_dim, 256, (len(self.network)*hidden_dim*2))
        self.mapping_network_app = CustomMappingNetwork(z_app_dim, 256, hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.transform_network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z_app, z_shape, ray_directions, **kwargs):
        frequencies_shape, phase_shifts_shape = self.mapping_network_shape(z_shape)
        frequencies_app, phase_shifts_app = self.mapping_network_app(z_app)  
        return self.forward_with_frequencies_phase_shifts(input, frequencies_shape, phase_shifts_shape, frequencies_app, phase_shifts_app, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies_shape, phase_shifts_shape, frequencies_app, phase_shifts_app, ray_directions, **kwargs):
        frequencies_shape = frequencies_shape*15 + 30
        frequencies_app = frequencies_app*15 + 30
        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies_shape[..., start:end], phase_shifts_shape[..., start:end])
        
        for layer in self.transform_network:
            x = layer(x)
            x = torch.sin(x)
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies_app, phase_shifts_app)
        rbg = self.color_layer_linear(rbg)

        return torch.cat([rbg, sigma], dim=-1)


##########################################################################################################

# encoder model for image
class inference(nn.Module):
    def __init__(self, z_shape_dim, z_app_dim, use_DDP=False):
        super(inference, self).__init__()
        self.mapping = nn.Linear(1000, 256)
        self.fc_shape_mean = nn.Linear(256, z_shape_dim)
        self.fc_shape_logvar = nn.Linear(256, z_shape_dim)
        self.fc_app_mean = nn.Linear(256, z_app_dim)
        self.fc_app_logvar = nn.Linear(256, z_app_dim)

        # initialize to 0
        self.fc_shape_mean.weight.data.zero_()
        self.fc_shape_logvar.weight.data.zero_()
        self.fc_app_mean.weight.data.zero_()
        self.fc_app_logvar.weight.data.zero_()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, use_sample=True): 
        assert len(x.shape) == 2
        h = self.mapping(x)
        h = F.relu(h)
        z_app_mean = self.fc_app_mean(h)
        z_app_logvar = self.fc_app_logvar(h)
        z_shape_mean = self.fc_shape_mean(h)
        z_shape_logvar = self.fc_shape_logvar(h)
        if use_sample:
            z_app_sample = z_app_mean + torch.exp(0.5 * z_app_logvar) * torch.randn_like(z_app_mean)
            z_shape_sample = z_shape_mean + torch.exp(0.5 * z_shape_logvar) * torch.randn_like(z_shape_mean)
        else:
            z_app_sample = z_app_mean
            z_shape_sample = z_shape_mean
        KL = torch.mean(0.5 * torch.sum(z_app_mean**2 + torch.exp(z_app_logvar) - 1 - z_app_logvar, dim=1)) +\
            torch.mean(0.5 * torch.sum(z_shape_mean**2 + torch.exp(z_shape_logvar) - 1 - z_shape_logvar, dim=1))
        return z_app_sample, z_shape_sample, KL

class pose_inference(nn.Module):
    # PoseNet based inference model, input image out put 4x4 rotation matrix
    # use vMF as prior
    
    def __init__(self, feat_dim=2048, radius=1.3, use_DDP=False):
        
        super(pose_inference, self).__init__()
        print('pose inference model see radius {}'.format(radius))
        feature_extractor = models.resnet34(pretrained=False)
        if use_DDP:
            self.feature_extractor = nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        else:
            self.feature_extractor = feature_extractor
        self.theta_mean = nn.Linear(1000, 2)
        self.theta_var = nn.Linear(1000, 1)
        self.phi_mean = nn.Linear(1000, 2)
        self.phi_var = nn.Linear(1000, 1)
        init_modules = [feature_extractor.fc, self.theta_mean, self.theta_var, self.phi_mean, self.phi_var]
    
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

        self.t1 = torch.tensor([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,radius],
                [0,0,0,1]], dtype=torch.float).cuda(non_blocking=True)
        self.t2 = torch.tensor([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.float).cuda(non_blocking=True)
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        

    def forward(self, x, use_sample = True, return_angle=False):
        x = self.normalize(x)
        pose = None
        h = self.feature_extractor(x)
        theta_mean = F.normalize(self.theta_mean(h), p=2, dim=1)
        phi_mean = F.normalize(self.phi_mean(h), p=2, dim=1)
        theta_var = F.softplus(self.theta_var(h)) + 1
        phi_var = F.softplus(self.phi_var(h) + 1)

        q_theta = VonMisesFisher(theta_mean, theta_var)
        q_phi = VonMisesFisher(phi_mean, phi_var)

        if use_sample:
            theta_sample = q_theta.rsample()
            phi_sample = q_phi.rsample()
        else:
            theta_sample = theta_mean
            phi_sample = phi_mean

        s_t = theta_sample[:, 0]
        c_t = theta_sample[:, 1]
        s_p = phi_sample[:, 0]
        c_p = phi_sample[:, 1]

        rot_phi = torch.reshape(torch.stack([
                    torch.ones_like(s_p), torch.zeros_like(s_p), torch.zeros_like(s_p), torch.zeros_like(s_p),
                    torch.zeros_like(s_p), c_p, -s_p, torch.zeros_like(s_p),
                    torch.zeros_like(s_p), s_p, c_p, torch.zeros_like(s_p),
                    torch.zeros_like(s_p), torch.zeros_like(s_p), torch.zeros_like(s_p), torch.ones_like(s_p)
            ], dim=1), (-1, 4, 4))

        rot_theta = torch.reshape(torch.stack([
                c_t, torch.zeros_like(s_t), -s_t, torch.zeros_like(s_t),
                torch.zeros_like(s_t), torch.ones_like(s_t), torch.zeros_like(s_t), torch.zeros_like(s_t),
                s_t, torch.zeros_like(s_t), c_t, torch.zeros_like(s_t),
                torch.zeros_like(s_t), torch.zeros_like(s_t), torch.zeros_like(s_t), torch.ones_like(s_t)
            ], dim=1), (-1, 4, 4)) 

        t2_tile = self.t2.unsqueeze(0).repeat(len(theta_sample), 1, 1)
        pose = torch.matmul(t2_tile, torch.matmul(rot_theta, torch.matmul(rot_phi, self.t1)))

        # calculate KL of posterior distribution with a uniform(on sphere) prior distribution equal to the negative entropy of the posterior distribution plus a constant 
        KL_theta = -q_theta.entropy().mean()
        KL_phi = -q_phi.entropy().mean()
        KL_pose = KL_theta + KL_phi


        if return_angle:
            _s_t = s_t.clone().detach().cpu().numpy()
            _c_t = c_t.clone().detach().cpu().numpy()
            _s_p = s_p.clone().detach().cpu().numpy()
            _c_p = c_p.clone().detach().cpu().numpy()
            theta = np.arctan2(_s_t, _c_t)
            phi = np.arctan2(_s_p, _c_p)

            return pose, KL_pose, theta, phi, h
        else:
            return pose, KL_pose, h


        


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1).to(c2w.get_device(), non_blocking=True)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    u = u.to(cdf.get_device(), non_blocking=True)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1).to(inds.get_device(), non_blocking=True), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples