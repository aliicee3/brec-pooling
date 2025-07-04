# Expressive Pooling for Graph Neural Networks

This repository contains the code for the paper "Expressive Pooling for Graph Neural Networks". 
Our implementation is built into the [BREC framework](https://github.com/GraphPKU/BREC). Please follow their instructions to install all necessary dependencies and download the BREC dataset.


To replicate the results presented in our paper, run 'run.sh' or execute the commands within that script. Options are the following:

* --POOLING={none,xp,edge_pool,topk,sag,asa,diff_pool,cluster,clique,curv} 
* --dataset={BREC_v3}