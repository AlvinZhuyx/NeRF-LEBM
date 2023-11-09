# NeRF-LEBM
This is the code for paper "Likelihood-Based Generative Radiance Field with Latent Space Energy-Based Model for 3D-Aware Disentangled Image Representation".

The Carla_MCMC_with_pose is for MCMC inference with ground truth pose,  Carla_amortize_inference is for amortized inference with ground truth pose, Carla_amortized_without_pose is for armortized inference without ground truth pose.

Some codes adapted from [this](https://github.com/yenchenlin/nerf-pytorch) awesome pytorch implementation of NeRF  
For implementing the experiments for training without ground truth camera pose, we use [this](https://github.com/nicola-decao/s-vae-pytorch) fabulous implementation of S-VAE. 
