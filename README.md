# Expressive Pooling for Graph Neural Networks

This repository contains the official implementation of our paper:

**"Expressive Pooling for Graph Neural Networks"**
Published in *Transactions on Machine Learning Research (2025)*
[OpenReview link](https://openreview.net/forum?id=xGADInGWMt)

Pooling layers are widely used in graph-level tasks to reduce graph size and capture hierarchical structure. Our work provides the **first formal proof** that under specific conditions, pooling operations which **decrease the size of the computational graphs** can **increase the expressive power** of Message Passing Neural Networks (MPNNs) — without modifying the message-passing scheme itself. We identify two sufficient conditions and implement several pooling strategies to empirically validate our theory.

## Project Overview

This implementation extends the [BREC framework](https://github.com/GraphPKU/BREC) by incorporating our proposed XP and comparative pooling methods for evaluating the expressive power of GNNs on challenging graph pairs from the BREC dataset.

**Newly added components**:

```
Pooling/
├── run.sh                       # Script to run the experiments
├── test_BREC.py                 # Evaluation pipeline for BREC dataset
├── BRECDataset_v3.py            # Updated wrapper for BREC v3 dataset
├── Data/                        # Include the dataset here
└── pooling/                     # Pooling methods and model implementations
    ├── XPooling.py              # XP: our proposed minimal expressivity-increasing pooling
    ├── cliquepooling.py         # Clique-based pooling (CliquePool)
    ├── ClusterPooling.py        # Edge score threshold-based pooling (ClusterPool)
    ├── curvpooling.py           # Curvature-based pooling (CurvPool)
    └── model.py                 # GNN model composition and pooling integration
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/aliicee3/brec-pooling
   cd brec-pooling
   ```

2. Follow the [BREC framework instructions](https://github.com/GraphPKU/BREC) to install dependencies and download the dataset.

3. Make sure the `BREC_v3` dataset is available in the expected location (Pooling/Data/raw/).

## Running Experiments

You can reproduce the results from our paper with:

```bash
bash run.sh
```

Or run experiments manually using:

```bash
python test_BREC.py --POOLING={none,xp,edge_pool,topk,sag,asa,diff_pool,cluster,clique,curv} --dataset=BREC_v3
```

The `--POOLING` option selects the pooling strategy.

## Implemented Pooling Methods

We provide and compare the following pooling operators:

- `xp`: Our proposed minimal and expressive pooling operator
- `edge_pool`: Iterative edge contraction based on edge scores
- `cluster`: Connected components based on edge score thresholds
- `clique`: Maximal clique pooling (computationally expensive)
- `curv`: Graph curvature-based pooling
- Baselines: `topk`, `sag`, `asa`, `diff_pool`, and `none`

See Section 5 of the [paper](https://openreview.net/forum?id=xGADInGWMt) for theoretical background and discussion.

## Evaluation on BREC

We evaluate all pooling methods on the [BREC dataset](https://github.com/GraphPKU/BREC), which consists of challenging pairs of **WL-indistinguishable but non-isomorphic graphs**. We report the number of such pairs each method successfully distinguishes — this serves as a measure of the method's **empirical expressivity**.

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{lachi2025expressive,
  title={Expressive Pooling for Graph Neural Networks},
  author={Lachi, Veronica and Moallemy-Oureh, Alice and Roth, Andreas and Welke, Pascal},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=xGADInGWMt}
}
```

## Contact

For questions or issues, please contact the authors via their institutional emails (see paper) or open an issue on this GitHub repository.


## Baselines

See the [BREC github repository](https://github.com/GraphPKU/BREC) for more details on baseline results reproduction. We include a clone of the respective directories for the architectures in the forked repository:

| Baseline          | Directory                           |
| ----------------- | ----------------------------------- |
| NGNN              | NestedGNN                           |
| DS-GNN            | SUN                                 |
| DSS-GNN           | SUN                                 |
| SUN               | SUN                                 |
| PPGN              | ProvablyPowerfulGraphNetworks_torch |
| GNN-AK            | GNNAsKernel                         |
| DE+NGNN           | NestedGNN                           |
| KP-GNN            | KP-GNN                              |
| KC-SetGNN         | KCSetGNN                            |
| I$^2$-GNN         | I2GNN                               |
| GSN               | GSN                                 |
| Graphormer        | Graphormer                          |
| OSAN              | OSAN                                |
| $\delta$-LGNN(SparseWL) | SparseWL                      |
| SWL               | SWL                                 |
| DropGNN           | DropGNN                             |
| Non-GNN Baselines | Non-GNN                             |
| Your Own GNN      | Base                                |

