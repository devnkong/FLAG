# FLAG: Adversarial Data Augmentation for Graph Neural Networks

This is the official repo for the paper [FLAG: Adversarial Data Augmentation for Graph Neural Networks](https://arxiv.org/abs/2010.09891).

**TL;DR:** FLAG augments node features to generalize GNNs on both node and graph classification tasks.

### Highlights

- **Simple**, adding just a dozen lines of code
- **Genera**l, applicable to any GNN baseline
- **Versatile**, working on both node and graph classification tasks
- **Scalable**, minimum memory overhead, working on the original infrastructure

## Experiments

To reproduce experimental results for **DeeperGCN**, visit [here](https://github.com/devnkong/FLAG/tree/main/deep_gcns_torch/examples/ogb).

Other baselines including **GCN**, **GraphSAGE**, **GAT**, **GIN**, **MLP**, etc. are available [here](https://github.com/devnkong/FLAG/tree/main/ogb).

To view the empirical performance of FLAG, please visit the Open Graph Benchmark [Node](https://ogb.stanford.edu/docs/leader_nodeprop/) and [Graph](https://ogb.stanford.edu/docs/leader_graphprop/) classification leaderboards.

### Requirements

- ogb=1.2.3
- torch-geometric=1.6.1
- torch=1.5.0

## Citing FLAG

If you find FLAG useful, please cite our paper.

```
@misc{kong2020flag,
      title={FLAG: Adversarial Data Augmentation for Graph Neural Networks}, 
      author={Kezhi Kong and Guohao Li and Mucong Ding and Zuxuan Wu and Chen Zhu and Bernard Ghanem and Gavin Taylor and Tom Goldstein},
      year={2020},
      eprint={2010.09891},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
