# Robust Optimization as Data Augmentation for Large-scale Graphs

This is the official repo for the paper [Robust Optimization as Data Augmentation for Large-scale Graphs](https://arxiv.org/abs/2010.09891), accepted at CVPR2022.

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

- ogb>=1.2.3
- torch-geometric>=1.6.1
- torch>=1.5.0

## Citing FLAG

If you find FLAG useful, please cite our paper.

```
@misc{https://doi.org/10.48550/arxiv.2010.09891,
  doi = {10.48550/ARXIV.2010.09891},
  url = {https://arxiv.org/abs/2010.09891},
  author = {Kong, Kezhi and Li, Guohao and Ding, Mucong and Wu, Zuxuan and Zhu, Chen and Ghanem, Bernard and Taylor, Gavin and Goldstein, Tom},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Robust Optimization as Data Augmentation for Large-scale Graphs},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
