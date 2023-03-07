#!/bin/bash
#SBATCH --partition=SCSEGPU_UG #SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name=MyJob #SBATCH --output=output_%x_%j.out #SBATCH --error=error_%x_%j.err
module load anaconda 
source home/FYP/fzhao006/.conda/envs/vitvae/bin/activate
source activate vitvae
conda env list
python --version
python run.py examples/blackened/config.ini