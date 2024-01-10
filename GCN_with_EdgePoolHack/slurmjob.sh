#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

source ~/.bashrc

conda deactivate
conda activate brec-env-server

/usr/bin/time --append --output=timing.info --format='%C\nuser: %U\nsys: %S\nwall: %E\n\n' python ~/brec-pooling/GCN_with_EdgePoolHack/test_BREC.py --POOLING edge_pool_base
/usr/bin/time --append --output=timing.info --format='%C\nuser: %U\nsys: %S\nwall: %E\n\n' python ~/brec-pooling/GCN_with_EdgePoolHack/test_BREC.py --POOLING edge_pool

conda deactivate
