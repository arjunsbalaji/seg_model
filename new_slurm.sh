#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=23:45:00
#SBATCH --output=octrc%j.out
#SBATCH --partition=gpuq
#SBATCH --constraint=p100
#SBATCH --account=pawsey0271
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --mail-type=END
#SBATCH --mail-user=21713337@student.uwa.edu.au

module load shifter
 

cd /group/pawsey0271/abalaji/projects/oct_ca_seg/seg_model/src
  

srun --export=all -n 1 shifter run adeytown75/pytorch:0.41-cuda9-cudnn7-devel-scikitimage python main.py


