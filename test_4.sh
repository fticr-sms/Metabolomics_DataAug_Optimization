#!/bin/sh

#SBATCH --job-name=GAN_Example
#SBATCH --output=output4.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=500
#SBATCH --account=cpca_a0_467905_2021

module purge
module load  OpenMPI/3.1.3-GCC-8.2.0-2.31.1


SECONDS=0

srun python GAN_script_example1.py

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


