This folder contains the sample code of NeRF-LEBM on Carla dataset without ground truth camera pose. We use armortized inference for both latent vectors and camera pose.

1. Install the environment:
To run the code, we recommend to follow the environment installation instruction of the official code of GRAF，the github link is here https://github.com/autonomousvision/graf
Also, in order for inference pose, we use the implementation of S-VAE, the github link is here https://github.com/nicola-decao/s-vae-pytorch Please install this package following its instructions.

2. Download the dataset:
Please download the dataset from the official github of GRAF, again the github link is here https://github.com/autonomousvision/graf 
Please download the image files and pose files, put them under path ./data/carla/imgs and ./data/carla/poses. 
In this setting, the model will actually discard the ground truth information during training. But we keep the same dataset API here, which keep consistency with other setting and enable to do testing.

3. Run the code, the code is designed to be run on multiple gpus parallelly, in our experiment, we use 2 gpus.
One can run the code with the following command: 
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=13477 train_nerf_lebm_nopose.py

4. The code will save its output to ./logs/carla_no_pose. It will save checkpoint under ./logs/carla_no_pose/ckpts and synthesis images in ./logs/carla_no_pose/imgs

5. Pre-trained checkpoint can be downloaded here https://drive.google.com/file/d/1buf-jlEFRmnm37z1Z_x4VkYczAj2uJ_R/view?usp=share_link
