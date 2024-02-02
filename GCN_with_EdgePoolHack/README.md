# Graph Pooling Provably Improves Expressivity

This repository contains the code for the paper "Graph Pooling Provably Improves Expressivity". 
Our implementation is built into the [BREC framework](https://github.com/GraphPKU/BREC).
We wrote a new DataLoader for the [RIGID dataset](https://www.lics.rwth-aachen.de/cms/LICS/Forschung/Publikationen/~rtok/Benchmark-Graphs/).


To replicate the results presented in our paper, run 'test_BREC.py' with the corresponding arguments:

* --POOLING={'xp','edge_pool','topk','sag','asa','diff_pool','none'} 
* --dataset={BREC_v3,RIGID} 
* --CONV_TYPE={gin,gcn,gat}
* --SEED={1,2,2023}

* For our proposed XP, we have three different configurations:
    * --MERGE={combine,single,max}

The computed statistics of a run can be found in the file 'results.tex' in a newly created folder with the prefix 'results_'.