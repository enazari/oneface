#!/bin/bash
#SBATCH --account=def-pbranco
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G            # memory for the entire job across all cores
#SBATCH --time=00-04:00      # time (DD-HH:MM)
#SBATCH --output=out_%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=csnazari@gmail.com   # Email to which notifications will be $



module load python/3.7
source ../../../env_masterface/bin/activate
cd ..
python main_Pset_Nset_V4.py --config_name=master1_seed0

