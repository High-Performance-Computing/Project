#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 1 
#SBATCH -c 8 
#SBATCH --gres=gpu:4 
#SBATCH -N 20 
#SBATCH --array=1-20 
#SBATCH -o output_file.txt 

module load python/3.8.5-fasrc01
module load cuda/11.0.3-fasrc01
module load cudnn/8.0.4.30_cuda11.0-fasrc01
source activate python3


srun -n 1 --gres=gpu:4 wandb agent davidassaraf/HP_tuning/9ixmaczy

