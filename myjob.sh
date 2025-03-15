#!/bin/bash
#SBATCH -J fcpflowtraining
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

# 1. Load Python module
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# 2. Activate your virtual env 
source $HOME/coopwitholga/bin/activate

# 3. Run your script
python /gpfs/home4/wxia/coopwitholga/comparative_analysis_for_data_agumentaion/data_augmentation/FCPFlow/main_fcpflow.py

