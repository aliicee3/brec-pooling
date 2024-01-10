#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=05:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

source ~/.bashrc

conda deactivate
conda activate brec-env-server

for hidden_dim in 8 16 32 64; do

    /usr/bin/time --append --output=timing.info --format='%C\nuser: %U\nsys: %S\nwall: %E\n\n' \
    python ~/brec-pooling/GCN_with_EdgePoolHack/test_BREC.py \
        --POOLING edge_pool_base \
        --HIDDEN_DIM $hidden_dim
    cat result_BREC_edge_pool_base/result_show.txt >> timing.info

    /usr/bin/time --append --output=timing.info --format='%C\nuser: %U\nsys: %S\nwall: %E\n\n' \
    python ~/brec-pooling/GCN_with_EdgePoolHack/test_BREC.py \
        --POOLING edge_pool \
        --HIDDEN_DIM $hidden_dim
    cat result_BREC_edge_pool_base/result_show.txt >> timing.info

done

conda deactivate
