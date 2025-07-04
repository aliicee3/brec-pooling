#!/usr/bin/zsh

# Run the grid search for all dataset and all methods
for dataset in BREC_v3; do
  for pooling in {xp,edge_pool,topk,none,sag,asa,diff_pool,cluster,clique,curv}; do
    for num_blocks in 3; do
      for num_layers in 4; do
        out_path=new_${pooling}_${dataset}_${num_blocks}_${num_layers}
        args="--POOLING=$pooling --DATASET $dataset --CONV_TYPE gin --NUM_BLOCKS=$num_blocks --NUM_LAYERS=$num_layers --PATH=$out_path"
        python test_BREC.py ${args}
      done
    done
  done
done