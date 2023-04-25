import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torch.multiprocessing as mp
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *
import pytorch_fid_wrapper as pfw
from ebm import EBM
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

DEBUG = False
activation = 'sigmoid'

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    
    def ret(inputs, z_app, z_shape, ray_direction):
        return torch.cat([fn(input=inputs[i:i+chunk], z_app=z_app[i: i+chunk], z_shape=z_shape[i: i+chunk],  ray_directions=ray_direction[i: i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, z_app=None, z_shape=None, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    # inputs should has shape [N * N_ray_per_images, N_samples, 3]
    # z should has shape [N * N_ray_per_images, z_dim]
    """
    if len(z_app) != len(inputs):
        print(z_app.shape, inputs.shape)
    assert len(z_app) == len(inputs) 
    #print(z_app.shape, inputs.shape, z_shape.shape)
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    z_app = torch.reshape(torch.unsqueeze(z_app, 1).repeat((1, inputs.shape[1], 1)), (-1, z_app.shape[-1]))
    z_shape = torch.reshape(torch.unsqueeze(z_shape, 1).repeat((1, inputs.shape[1], 1)), (-1, z_shape.shape[-1]))

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

    outputs_flat = batchify(fn, netchunk)(inputs_flat, z_app, z_shape, input_dirs_flat)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify_rays(rays_flat, chunk=1024*32, z_app=None, z_shape=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    #print('batchify_rays', rays_flat.shape, z.shape)
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], z_app=z_app[i:i+chunk], z_shape=z_shape[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, z_app=None, z_shape=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):

    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [N, N_rays_per_imag, 2, 3]. Ray origin and direction for
        each example in batch.
      z: latent variables  with shape (N, z_app_dim); (N, z_shape_dim)
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_o = torch.reshape(rays_o, (1, -1, 3))
        rays_d = torch.reshape(rays_d, (1, -1, 3))
    else:
        # use provided ray batch
        rays_o, rays_d = rays[:, :, 0, :], rays[:, :, 1, :]
    #print(rays_o.shape, rays_d.shape)
    N_rays_per_image = int(rays_o.shape[1])
    z_app_dim = int(z_app.shape[-1])
    z_shape_dim = int(z_shape.shape[-1])
    
    z_app = torch.reshape(torch.unsqueeze(z_app, 1).repeat((1, N_rays_per_image, 1)), (-1, z_app_dim)) # (N * N_rays_per_image, z_app_dim)
    z_shape = torch.reshape(torch.unsqueeze(z_shape, 1).repeat((1, N_rays_per_image, 1)), (-1, z_shape_dim)) # (N * N_rays_per_image, z_app_dim)

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)


        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    #print('ray shape', rays.shape)
    all_ret = batchify_rays(rays, chunk, z_app=z_app, z_shape=z_shape, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, z_app, z_shape, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, global_step=0, type='sample', rank=0, saveimg=True):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        t = time.time()
        if 'sample' in type or 'novel_view' in type:
            rgb, disp, acc, extras = render(H, W, K, z_app=z_app, z_shape=z_shape, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        else:
            assert len(z_app) == len(render_poses)
            rgb, disp, acc, extras = render(H, W, K, z_app=z_app[i:i+1], z_shape=z_shape[i:i+1], chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        if 'rgb0' in extras:
            rgbs.append(extras['rgb0'])
        else:
            rgbs.append(rgb)
        disps.append(disp)

    rgbs = torch.cat(rgbs, 0)
    disps = torch.cat(disps, 0)
    N_img = rgbs.shape[0]
    rgbs = torch.reshape(rgbs, (len(rgbs), H, W, 3)).permute([0, 3, 1, 2])
    disps = torch.reshape(disps, (len(rgbs), H, W, 1)).permute([0, 3, 1, 2])
    if saveimg:
        rgbs = torchvision.utils.make_grid(rgbs, nrow=min(11, N_img))
        torchvision.utils.save_image(rgbs.clone(), os.path.join(savedir, '{}_{:06d}_{}_rgb.png'.format(type, global_step, rank)))
        disps = torchvision.utils.make_grid((disps - disps.min()) / (disps.max() - disps.min()), nrow=min(11, N_img))
        torchvision.utils.save_image(disps.clone(), os.path.join(savedir, '{}_{:06d}_{}_density.png'.format(type, global_step, rank)))

    return rgbs, disps

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    model = TALLSIREN(input_dim=3, z_shape_dim=args.z_shape_dim, z_app_dim=args.z_app_dim, hidden_dim=256, output_dim=1)
    model.cuda()
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = TALLSIREN(input_dim=3, z_app_dim=args.z_app_dim, z_shape_dim=args.z_shape_dim, hidden_dim=256, output_dim=1)
        model_fine.cuda()
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn, z_app, z_shape: run_network(inputs, viewdirs, network_fn,
                                                                netchunk=args.netchunk,
                                                                z_app = z_app,
                                                                z_shape = z_shape)
    
    ebm_shape = EBM(args.z_shape_dim, 128, nout=1, prior_type=args.ptype, act_type=args.ebm_act).cuda()
    ebm_app = EBM(args.z_app_dim, 64, nout=1, prior_type=args.ptype_app, act_type=args.ebm_act).cuda()

    inference_model = inference(z_shape_dim=args.z_shape_dim, z_app_dim = args.z_app_dim, use_DDP=True).cuda()
    pose_estimator = pose_inference(use_DDP=True, radius=args.radius).cuda()

    grad_vars_inference = list(inference_model.parameters())
    grad_vars_pose = list(pose_estimator.parameters())

    grad_vars_ebm, optimizer_ebm = None, None
    if 'ebm' in args.ptype:
        grad_vars_ebm = list(ebm_shape.parameters()) + list(ebm_app.parameters())
    

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    optimizer_inference = torch.optim.Adam(params=grad_vars_inference, lr=args.lrate_infer, betas=(0.9, 0.999))
    optimizer_pose = torch.optim.Adam(params=grad_vars_pose, lr=args.lrate_pose, betas=(0.9, 0.999))
    if 'ebm' in args.ptype:
        optimizer_ebm = torch.optim.Adam(params=grad_vars_ebm, lr=args.lrate_ebm, betas=(0.9, 0.999))


    start = 0
    basedir = args.basedir
    expname = args.expname
    
    # Apply DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    if model_fine is not None:
        model_fine = nn.parallel.DistributedDataParallel(model_fine, device_ids=[args.local_rank], output_device=args.local_rank)
    ebm_app = nn.parallel.DistributedDataParallel(ebm_app, device_ids=[args.local_rank], output_device=args.local_rank)
    ebm_shape = nn.parallel.DistributedDataParallel(ebm_shape, device_ids=[args.local_rank], output_device=args.local_rank)
    inference_model = nn.parallel.DistributedDataParallel(inference_model, device_ids=[args.local_rank], output_device=args.local_rank)
    pose_estimator = nn.parallel.DistributedDataParallel(pose_estimator, device_ids=[args.local_rank], output_device=args.local_rank)
    
    ##########################
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, 'ckpts', f) for f in sorted(os.listdir(os.path.join(basedir, expname, 'ckpts'))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location={'cuda:%d' % 0: 'cuda:%d' % args.local_rank})

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_inference.load_state_dict(ckpt['optimizer_inference_state_dict'])
        optimizer_pose.load_state_dict(ckpt['optimizer_pose_state_dict'])
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        pose_estimator.load_state_dict(ckpt['pose_estimator_state_dict'])
        inference_model.load_state_dict(ckpt['inference_model_state_dict'])
        if 'ebm' in args.ptype:
            ebm_shape.load_state_dict(ckpt['ebm_shape_state_dict'])
            ebm_app.load_state_dict(ckpt['ebm_app_state_dict'])
            optimizer_ebm.load_state_dict(ckpt['optimizer_ebm_state_dict'])
    ##########################
    
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, grad_vars_ebm, grad_vars_inference, grad_vars_pose, optimizer, optimizer_ebm, optimizer_pose, optimizer_inference, ebm_app, ebm_shape, inference_model, pose_estimator

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.get_device())], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    if activation == 'sigmoid':
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    else:
        rgb = 0.5 * (torch.tanh(raw[...,:3]) + 1.0)
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.get_device()), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(ray_batch,
                z_app,
                z_shape,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(ray_batch.get_device())
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(ray_batch.get_device())

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(ray_batch.get_device())
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    raw = network_query_fn(pts, viewdirs, network_fn, z_app=z_app, z_shape=z_shape)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn, z_app=z_app, z_shape=z_shape)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, type=str, default='./config/configs.txt',
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/carla',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--z_app_dim", type=int, default=128,
                        help='appearance latent variable size')
    parser.add_argument("--z_shape_dim", type=int, default=128,
                        help='appearance latent variable size')
    parser.add_argument("--im_size", type=int, default=128,
                        help='image size')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (in term of images number)')
    parser.add_argument("--n_views", type=int, default=2,
                        help='number of views per object')
    parser.add_argument("--lrate", type=float, default=1e-4, 
                        help='learning rate') # original 5e-4
    parser.add_argument("--lrate_pose", type=float, default=1e-4, 
                        help='learning rate') 
    parser.add_argument("--lrate_infer", type=float, default=1e-4, 
                        help='learning rate') 
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--lrate_ebm",type=float, default=2e-5,
                        help='learning rate for ebm')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_sample",     type=int, default=1000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=1000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_test",   type=int, default=1000,
                        help='frequency of test on novel objects')

    

    # Langevin dynamtic options
    parser.add_argument('--e_l_steps', type=int, default=60, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=False, type=bool, help='noise term of langevin')
    parser.add_argument('--ptype', default='ebm', type=str, help='whether use ebm or gaussain prior')
    parser.add_argument('--ptype_app', default='ebm', type=str, help='whether use ebm or gaussain prior')
    parser.add_argument('--ebm_act', default='relu', type=str, help='activation used in latent ebm')

    parser.add_argument('--g_llhd_sigma', type=float, default=1.0, help='prior of factor analysis')
    parser.add_argument('--g_llhd_sigma_start', type=float, default=1.0, help='prior of factor analysis')
    parser.add_argument('--warm_up_steps', type=int, default=20000, help='prior of factor analysis')
    parser.add_argument('--g_l_steps', type=int, default=60, help='number of langevin steps')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of langevin')
    parser.add_argument('--g_l_with_noise', default=False, type=bool, help='noise term of langevin')
    parser.add_argument('--n_epochs', default=100, type=int, help = 'total training epochs')
    
    # control the gaussain prior of appearance and shape so that the shape factor and explain more 
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--update_steps', type=int, default=3, help = 'number of parameter updates between inference z')
    # Multi-GPU args
    parser.add_argument('--local_rank', type=int, metavar='N', help='local')
    parser.add_argument('--train', type=bool, default=False, metavar='N', help='local')

    return parser

def sample_p0(N, nz):
    return torch.randn(N, nz).float().cuda(non_blocking=True)

def train(args):
    dist.init_process_group(backend='nccl')
    device = torch.device('cuda', args.local_rank)
    print('running process on ', device, device == torch.device('cuda', 0))
    torch.cuda.set_device(device)

    K = None
    # Dataloader
    if args.dataset_type == 'carla':
        from load_carla_persist import ImageDataset
        near = 7.5
        far = 12.5
        args.radius = 10
        dataset = ImageDataset(data_dirs=args.datadir,  im_sz=args.im_size)
        args.data_number = len(dataset)
        print('total number of training data ', args.data_number)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0, 
                        pin_memory=True, sampler=train_sampler)    
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = dataset.im_sz, dataset.im_sz, dataset.focal
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    savedir = os.path.join(basedir, expname, 'imgs')
    os.makedirs(os.path.join(basedir, expname, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'ckpts'), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, grad_vars_ebm, grad_vars_inference, grad_vars_pose, optimizer, optimizer_ebm, optimizer_pose, optimizer_inference, ebm_app, ebm_shape, inference_model, pose_estimator\
     = create_nerf(args)
    
    
    global_step = start + 1

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Prepare raybatch tensor if batching random rays
    N_rand_per_img = min(args.N_rand // (args.batch_size * args.n_views), H * W)


    print('Begin training')

    # Summary writers
    time0 = time.time()
    start = start // len(dataloader)  + 1

    #for i in trange(start, args.n_epochs):
    for i in range(start + 1, args.n_epochs):
        train_sampler.set_epoch(i)
        for imgs, _, idx in dataloader:
            for param_group in optimizer_inference.param_groups:
                param_group['lr'] = min(max(0, global_step - args.warm_up_steps) / 10000., 1.0) * args.lrate_infer

            imgs = imgs.cuda(non_blocking=True) # imgs should in shape (N, n_views, 3, H, W)
            poses_infer, KL_pose, theta_pred, phi_pred, pose_h = pose_estimator(torch.reshape(imgs, (-1, 3, H, W)), use_sample=True, return_angle=True)
            
            b_sz = imgs.shape[0]
            imgs = torch.reshape(imgs, (b_sz * args.n_views, 3, H, W))
            imgs_clone = imgs.clone().detach()
            idx = idx.cuda(non_blocking=True)
            idx = torch.squeeze(idx)

            # img shape (N, 3, H, W)
            rays = torch.stack([torch.stack(get_rays(H, W, K, p), 0) for p in poses_infer[:,:3,:4]], 0) # rays should have shape (N, ro+rd,H, W, 3)
            imgs = torch.unsqueeze(imgs.permute([0, 2, 3, 1]), 1) # img shape (N, rgb, H, W, 3)
            rays_rgb = torch.cat([rays, imgs], 1) #(N, ro+rd+rgb, H, W, 3)
            rays_rgb = rays_rgb.permute([0, 2, 3, 1, 4]) #(N, H, W, ro+rd+rgb, 3)
            rays_rgb = torch.reshape(rays_rgb, (-1, H*W, 3, 3))
            
            # sample rays
            batch_rays_rgb = []
            for ray_rgb in rays_rgb:
                index = np.random.choice(H*W, N_rand_per_img, replace=False)
                batch_rays_rgb.append(ray_rgb[index])
            
            batch_rays_rgb = torch.stack(batch_rays_rgb, 0) #(N, N_rand_per_img, ro+rd+rgb, 3)

            batch_rays, target_s = batch_rays_rgb[:, :, :2, :], batch_rays_rgb[:, :, 2, :]
            
            #####  Core optimization loop  #####
            # do posterior sample
            g_llhd_sigma = args.g_llhd_sigma_start + (args.g_llhd_sigma - args.g_llhd_sigma_start) * min(float(global_step) / args.warm_up_steps, 1.0) 
            for iii in range(args.update_steps):
                z_app, z_shape, KL_loss = inference_model(pose_h, use_sample=True)
                z_app = torch.mean(torch.reshape(z_app, (b_sz, args.n_views, args.z_app_dim)), dim=1)
                z_shape = torch.mean(torch.reshape(z_shape, (b_sz, args.n_views, args.z_shape_dim)), dim=1)
                
                z_app_tile = torch.reshape(torch.unsqueeze(z_app, dim=1).repeat(1, args.n_views, 1), (b_sz * args.n_views, args.z_app_dim))
                z_shape_tile = torch.reshape(torch.unsqueeze(z_shape, dim=1).repeat(1, args.n_views, 1), (b_sz * args.n_views, args.z_shape_dim))
                
                rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, z_app=z_app_tile, z_shape=z_shape_tile,
                                                        verbose=global_step < 10, retraw=True,
                                                        **render_kwargs_train)
                  
                en_pos_shape = torch.sum(torch.mean(ebm_shape(z_shape), dim=0))
                en_pos_app = torch.sum(torch.mean(ebm_app(z_app), dim=0))
                optimizer.zero_grad()
                optimizer_pose.zero_grad()
                optimizer_inference.zero_grad()
                img_loss = img2mse(rgb, target_s) 
                trans = extras['raw'][...,-1]
                
                loss = img_loss * 0.5 / (g_llhd_sigma ** 2) * args.N_rand / args.batch_size + KL_loss + KL_pose + en_pos_shape + en_pos_app
                psnr = mse2psnr(img_loss)

                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], target_s) 
                    loss = loss + img_loss0 * 0.5 / (g_llhd_sigma ** 2) * args.N_rand / args.batch_size
                    psnr0 = mse2psnr(img_loss0) 
                else:
                    img_loss0 = torch.Tensor([0.0])
                    psnr0 = torch.Tensor([0.0])

                loss.backward()
                optimizer.step()
                optimizer_inference.step()
                optimizer_pose.step()
            
            z_app = z_app.detach()
            z_shape = z_shape.detach()
            # update ebm
            if 'ebm' in args.ptype:
                optimizer_ebm.zero_grad()
            z_neg_shape = torch.autograd.Variable(args.sigma * sample_p0(50, args.z_shape_dim), requires_grad=True) # using more samples in estimate the expectation
            z_neg_app = torch.autograd.Variable(args.sigma * sample_p0(50, args.z_app_dim), requires_grad=True)
            e_l_steps = args.e_l_steps
            if 'ebm' in args.ptype:
                for _ in range(e_l_steps):
                    en = ebm_shape(z_neg_shape).sum() + 0.5 * torch.sum(z_neg_shape ** 2) + ebm_app(z_neg_app).sum() + 0.5 * torch.sum(z_neg_app ** 2)
                    grad_shape, grad_app = torch.autograd.grad(en, [z_neg_shape, z_neg_app])
                    z_neg_shape.data = z_neg_shape.data - 0.5 * args.e_l_step_size ** 2 * grad_shape
                    z_neg_app.data = z_neg_app.data - 0.5 * args.e_l_step_size ** 2 * grad_app
                    
                    if args.e_l_with_noise:
                        z_neg_shape.data = z_neg_shape.data + args.e_l_step_size * torch.randn_like(z_neg_shape).data
                        z_neg_app.data = z_neg_app.data + args.e_l_step_size * torch.randn_like(z_neg_app).data
                        
                z_neg_app = z_neg_app.detach()
                z_neg_shape = z_neg_shape.detach()
                optimizer_ebm.zero_grad()
                en_pos_shape = torch.sum(torch.mean(ebm_shape(z_shape.detach()), dim=0))
                en_neg_shape = torch.sum(torch.mean(ebm_shape(z_neg_shape.detach()), dim=0))
                en_pos_app = torch.sum(torch.mean(ebm_app(z_app.detach()), dim=0))
                en_neg_app = torch.sum(torch.mean(ebm_app(z_neg_app.detach()), dim=0))
                
                loss_e_shape = en_pos_shape - en_neg_shape
                loss_e_app = en_pos_app - en_neg_app
                loss_e = loss_e_shape + loss_e_app 
                loss_e.backward()
                optimizer_ebm.step()
            else:
                with torch.no_grad():
                    en_pos_app = torch.sum(torch.mean(ebm_app(z_app.detach()), dim=0))
                    en_neg_app = torch.sum(torch.mean(ebm_app(z_neg_app), dim=0))
                    loss_e_app = en_pos_app - en_neg_app
                    en_pos_shape = torch.sum(torch.mean(ebm_shape(z_shape.detach()), dim=0))
                    en_neg_shape = torch.sum(torch.mean(ebm_shape(z_neg_shape), dim=0))
                    loss_e_shape = en_pos_shape - en_neg_shape
                    loss_e = loss_e_shape + loss_e_app
            
            ###############################################################################################################################################

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            new_lrate_pose = args.lrate_pose * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            
            for param_group in optimizer_pose.param_groups:
                param_group['lr'] = new_lrate_pose
            
            if 'ebm' in args.ptype:
                new_lrate_ebm = args.lrate_ebm * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer_ebm.param_groups:
                    param_group['lr'] = new_lrate_ebm
            ################################
            #####           end            #####

            # Rest is logging
            if True and (global_step % args.i_weights==0 and device == torch.device('cuda', 0)):

                path = os.path.join(basedir, expname, 'ckpts', '{:06d}.tar'.format(global_step))
                save_dict = {
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_pose_state_dict': optimizer_pose.state_dict(),
                    'optimizer_inference_state_dict': optimizer_inference.state_dict(),
                    'ebm_app_state_dict': ebm_app.state_dict(),
                    'ebm_shape_state_dict': ebm_shape.state_dict(),
                    'pose_estimator_state_dict': pose_estimator.state_dict(),
                    'inference_model_state_dict': inference_model.state_dict()
                }
                if 'network_fine' in render_kwargs_train.keys() and render_kwargs_train['network_fine'] is not None:
                    save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
                
                if optimizer_ebm is not None:
                    save_dict['optimizer_ebm_state_dict'] = optimizer_ebm.state_dict()
                torch.save(save_dict, path)
                print('Saved checkpoints at', path)

            if True and global_step % args.i_sample==0: #and device == torch.device('cuda', 0):
                # Render the image of the first sampled car
                # to test multiview generation, we input infered results 

                # render around the first observed image
                render_poses = dataset.get_render_poses(max_theta=theta_pred[0] + 90./180 * np.pi, min_theta=theta_pred[0] - 90./180, max_phi=phi_pred[0] + 40./180 * np.pi, min_phi=phi_pred[0] - 40./180 * np.pi, ntheta=11, nphi=11)
                with torch.no_grad():
                    render_path(render_poses, z_neg_app[:1, :], z_neg_shape[:1, :], hwf, K, args.chunk, render_kwargs_test, savedir=savedir, global_step=global_step, type='sample', rank = args.local_rank, saveimg=(device == torch.device('cuda', 0)))

                t_imgs = []
                t_disps = []
                with torch.no_grad():
                    for iii in range(5):
                        for jjj in range(5):
                            t_img, t_disp = render_path(render_poses[60:61, :, :], z_app[iii:iii+1, :], z_shape[jjj:jjj+1, :], hwf, K, args.chunk, render_kwargs_test, savedir=savedir, global_step=global_step, type='sample', rank = args.local_rank, saveimg=False)
                            t_imgs.append(t_img.detach().clone())
                            t_disps.append(t_disp.detach().clone())
                
                if device == torch.device('cuda', 0):
                    t_imgs = torch.cat(t_imgs, 0)
                    t_imgs = torchvision.utils.make_grid(t_imgs, nrow=5)
                    torchvision.utils.save_image(t_imgs.clone(), os.path.join(savedir, 'sample_inter_{}.png'.format(global_step)))
                    t_disps = torch.cat(t_disps, 0)
                    t_disps = torchvision.utils.make_grid((t_disps - t_disps.min()) / (t_disps.max() - t_disps.min()), nrow=5)
                    torchvision.utils.save_image(t_disps.clone(), os.path.join(savedir, 'sample_inter_{}_disps.png'.format(global_step)))
                    print('Done, sampling.')
            
            if True and global_step % args.i_testset==0:
                if device == torch.device('cuda', 0):
                    print('{} begin reconstruction'.format(device))
                poses_infer, _, theta_pred, phi_pred, pose_h = pose_estimator(imgs_clone, use_sample=False, return_angle=True)
                z_app, z_shape, KL_loss = inference_model(pose_h, use_sample=False)
                
                # Reconstruct image
                with torch.no_grad():
                    z_app = torch.mean(torch.reshape(z_app, (b_sz, args.n_views, args.z_app_dim)), dim=1)
                    z_shape = torch.mean(torch.reshape(z_shape, (b_sz, args.n_views, args.z_shape_dim)), dim=1)
                    z_app_tile = torch.reshape(torch.unsqueeze(z_app, dim=1).repeat(1, args.n_views, 1), (b_sz * args.n_views, args.z_app_dim))
                    z_shape_tile = torch.reshape(torch.unsqueeze(z_shape, dim=1).repeat(1, args.n_views, 1), (b_sz * args.n_views, args.z_shape_dim))
                    render_path(poses_infer, z_app_tile, z_shape_tile, hwf, K, args.chunk, render_kwargs_test, savedir=savedir, global_step=global_step, type='recons', rank = args.local_rank, saveimg=(device == torch.device('cuda', 0)))

                imgs = torch.squeeze(imgs)

                if len(imgs.shape) < 4:
                    imgs = torch.unsqueeze(imgs, 0)
                imgs = imgs.permute((0, 3, 1, 2))

                imgs_grid = torchvision.utils.make_grid(imgs, nrow=min(11, len(imgs)))
                if device == torch.device('cuda', 0):
                    torchvision.utils.save_image(imgs_grid.clone(), os.path.join(savedir, '{}_{:06d}_{}_train_ori.png'.format('recons', global_step, args.local_rank)))
                
                for iii in range(min(2, len(z_app))):
                    gen_imgs = []
                    gen_disps = []
                    render_poses = dataset.get_render_poses(max_theta=theta_pred[iii] + 90./180 * np.pi, min_theta=theta_pred[iii] - 90./180, max_phi=phi_pred[iii] + 40./180 * np.pi, min_phi=phi_pred[iii] - 40./180 * np.pi, ntheta=11, nphi=11)
                    with torch.no_grad():
                        rendered_imgs, disps = render_path(render_poses, z_app[iii: iii+1, :], z_shape[iii: iii+1, :], hwf, K, args.chunk, render_kwargs_test, savedir=savedir, global_step=global_step, type='novel_view', rank = args.local_rank, saveimg=False)
                    gen_imgs.append(rendered_imgs)
                    gen_disps.append(disps)
                    if device == torch.device('cuda', 0):
                        gen_imgs = torch.cat(gen_imgs, 0)  
                        gen_imgs_grid = torchvision.utils.make_grid(gen_imgs, nrow=min(11, len(gen_imgs)))
                        torchvision.utils.save_image(gen_imgs_grid.clone(), os.path.join(savedir, '{}_{:06d}_val_gen_ins{}.png'.format('recons', global_step,  iii)))
                        gen_disps = torch.cat(gen_disps, 0)  
                        gen_disps_grid = torchvision.utils.make_grid((gen_disps - gen_disps.min()) / (gen_disps.max() - gen_disps.min()), nrow=min(11, len(gen_imgs)))
                        torchvision.utils.save_image(gen_disps_grid.clone(), os.path.join(savedir, '{}_{:06d}_val_gen_ins{}_disps.png'.format('recons', global_step,  iii)))

            if  (True and global_step % args.i_print==0) and device == torch.device('cuda', 0):
                dt = time.time()-time0
                print(f"[TRAIN] Iter: {i} step {global_step} Time: {dt:.2f} Loss: {loss.item():.3f}  PSNR: {psnr.item():.2f} {psnr0.item():.2f}  KL_z: {KL_loss.item():.2f} KL_pose: {KL_pose.item():.2f} en_pos_shape: {en_pos_shape.item():.3f} en_neg_shape: {en_neg_shape.item():.3f} en_pos_app: {en_pos_app.item():.3f} en_neg_app: {en_neg_app.item():.3f} g_llhd_sigma: {g_llhd_sigma:.3f}")
                time0 = time.time()
        
            global_step += 1
    

if __name__=='__main__':
    torch.set_default_tensor_type('torch.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    args.activation = activation
    random.seed(args.local_rank + 1)
    np.random.seed(args.local_rank + 1)
    torch.manual_seed(args.local_rank + 1)
    torch.cuda.manual_seed_all(args.local_rank + 1)
    train(args)