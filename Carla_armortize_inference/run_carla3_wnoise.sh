#!/bin/bash
#SBATCH --job-name=cvae3_wn
#SBATCH --output=carla_vae3_wnoise.out
#SBATCH --error=carla_vae3_wnoise.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=4-23:59:00
#SBATCH --partition=1080Ti_slong

cd ~/nerf_ABP2
source ~/miniconda3/bin/activate
conda activate graf
srun python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=13677 run_nerf_mv2_vae_gaussian_carla3_wnoise.py 
