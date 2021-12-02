#!/bin/bash
#
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
python3 preprocessdata.py 5
python3 preprocessdata.py 20
# python3 preprocessdata.py 60