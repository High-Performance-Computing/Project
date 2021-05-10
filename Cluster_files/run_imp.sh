#!/bin/bash
#SBATCH -p gpu
#SBATCH -n 1 
#SBATCH -c 8 
#SBATCH --gres=gpu:4 
#SBATCH -N 20 
#SBATCH --array=1-20 
#SBATCH -o output_file_raph_%A_%a.out 
module load python


python masking.py -input "$(< mask/input_masks_raph_$SLURM_ARRAY_TASK_ID)"

