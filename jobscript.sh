#!/bin/bash

#SBATCH -N 1 
#SBATCH --ntasks-per-node=8 #number of cores per node
#SBATCH --time=2-00:00:00 
#SBATCH --job-name=ResShift_inference  #change name of ur job
#SBATCH --error=error   #change name the error file
#SBATCH --output=output  #change name of ur output file
#SBATCH --partition=gpu  #there are various partition. U can change various GPUs
#SBATCH --gres=gpu:2 #same as above
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=t22104@students.iitmandi.ac.in

# Load module
module load DL-Conda_3.7

# activate environment 
source /home/apps/DL/DL-CondaPy3.7/bin/activate ScSR
cd $SLURM_SUBMIT_DIR

python learn_dictionry.py