#!/bin/bash
#SBATCH --job-name=rlQuantumSwitch
#SBATCH --nodes=1
#SBATCH --cluster=htc
#SBATCH --ntasks-per-node=60
#SBATCH --time=0-12:00:00 
#SBATCH --mail-user=vik80@pitt.edu
#SBATCH --mail-type=END,FAIL
  
#Load python via LMOD 
module purge
module load gcc/8.2.0 python/anaconda3.10-2022.10
  
#Activate your environment 
source activate /ihome/kseshadreesan/vik80/Research_Fall23/python_envs/optimized_switch
  
#Run commands utilizing your loaded Python tool 
python main2.py

python crc-job-stats.py

