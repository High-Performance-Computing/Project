#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 4-10:00 #
#SBATCH -J LTH
#SBATCH -o tf_imagenet.out
#SBATCH -e tf_imagenet.err
#SBATCH --gres=gpu:4
#SBATCH --gpu-freq=high
#SBATCH --mem=100G

# --- Set up software environment ---
module load python
module load cuda/11.0.3-fasrc01
module load cudnn/8.0.4.30_cuda11.0-fasrc01
source activate python3

# --- Run the code ---
srun -n 1 --gres=gpu:4 python train_model.py
