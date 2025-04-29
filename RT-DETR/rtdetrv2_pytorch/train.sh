#!/bin/bash
#SBATCH --job-name=rtdetrv2
#SBATCH --partition=l40-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4             # 3 tasks = 3 processes = 3 GPUs
#SBATCH --gpus-per-node=4               # 3 GPUs per node
#SBATCH --mem=512G                       # Max allowed memory
#SBATCH --time=2-00:00:00                # 2 days
#SBATCH --output=slurm-%j.out            # Save Slurm logs per job
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=prakrut@cs.unc.edu

module load anaconda/2023.03

# Activate Conda environment
source ~/.bashrc
conda activate rt

# Navigate to your RT-DETR v2 project directory
cd /work/users/p/r/prakrut/RT-DETR/rtdetrv2_pytorch/

nvidia-smi

# Launch a dummy distributed script (small dry run)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r output/rtdetrv2_r18vd_120e_coco/checkpoint0049.pth --use-amp --seed=0 --summary-dir=summary