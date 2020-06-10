#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=15:00:00
#SBATCH --output=MLFLOW_D2%j.out
#SBATCH --partition=gpuq
#SBATCH --constraint=p100
#SBATCH --account=pawsey0271
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --mail-type=END
#SBATCH --mail-user=21713337@student.uwa.edu.au

module load singularity
export myRep=$MYGROUP/singularity/oct_ca 
export containerImage=$myRep/oct_ca_latest-fastai-skl-ski-mlflow-d2-opencv-coco.sif
export projectDir=$MYGROUP/projects

cd /group/pawsey0271/abalaji/projects/oct_ca_seg/seg_model/  

ulimit -s unlimited

export X_MEMTYPE_CACHE=n

srun --export=all -n 1 singularity exec -B $projectDir:/workspace --nv $containerImage python3 nbs/train.py 'd2' 90000 4 1 && python3 nbs/validate.py 'd2_pawsey' 0.9 1  && python3 validate.py 'd2_pawsey' 0.8 1  && python3 validate.py 'd2_pawsey' 0.7 1  && python3 validate.py 'd2_pawsey' 0.6 1  && python3 validate.py 'd2_pawsey' 0.5 1 

