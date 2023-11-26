#!/bin/sh

#SBATCH --job-name=ODint-nnodes-study
#SBATCH --output=ODint-nnodes-%A.%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1600
#SBATCH --account=cpca_a0_467905_2021
#SBATCH --array=0-3

module purge
module load  OpenMPI/3.1.3-GCC-8.2.0-2.31.1

echo "slurm job id is $SLURM_ARRAY_JOB_ID."
echo "slurm task id is $SLURM_ARRAY_TASK_ID."
echo "Executing on the machine:" $(hostname)

SECONDS=0

srun python ODint_nnodes_study_GAN.py

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


