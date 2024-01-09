#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

module load Python
python GCN_with_Edge_PoolHack/test_BREC.py --POOLING edge_pool_base 
python GCN_with_Edge_PoolHack/test_BREC.py --POOLING edge_pool