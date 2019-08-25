#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=05:45:00
#SBATCH --output=octrc%j.out
#SBATCH --partition=copyq
#SBATCH --account=pawsey0271
#SBATCH --export=NONE
#SBATCH --mail-type=END
#SBATCH --mail-user=21713337@student.uwa.edu.au

module load shifter
 

mv '/scratch/pawsey0271/abalaji/projects/oct_ca_seg/actual final data' /group/pawsey0271/abalaji/projects/oct_ca_seg/
  

