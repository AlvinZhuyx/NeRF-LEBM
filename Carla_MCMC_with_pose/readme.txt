This folder contains the sample code of NeRF-LEBM using MCMC inference on Carla dataset.

1. Install the environment:
To run the code, one can follow the environment installation instruction of the official code of GRAF，the github link is here https://github.com/autonomousvision/graf

2. Download the dataset:
Please download the dataset from the official github of GRAF, again the github link is here https://github.com/autonomousvision/graf 
Please download the image files and code files, put them under path ./data/carla/imgs and ./data/carla/poses

3. Run the code, the code is designed to be run on multiple gpus parallelly, in our experiment, we use 4 gpus.
One can run the code with the following command: 
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=13477 train_nerf_lebm.py

4. The code will save its output to ./logs/carla. It will save checkpoint under ./logs/carla/ckpts and synthesis images in ./logs/carla/imgs

5. Pretrained checkpoint can be downloaded here https://drive.google.com/file/d/1Uq3MGy6PQZ_8x2T2uY9AfSphAFTMlOjn/view?usp=sharing
