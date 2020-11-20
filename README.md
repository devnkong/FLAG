# FLAG: Adversarial Data Augmentation for Graph Neural Networks

This is the official repo for the paper [FLAG: Adversarial Data Augmentation for Graph Neural Networks](https://arxiv.org/abs/2010.09891).

Data augmentation helps neural networks generalize better, but it remains an open question how to effectively augment graph data to enhance the performance of GNNs (Graph Neural Networks). While most existing graph regularizers focus on augmenting graph topological structures by adding/removing edges, we offer a novel direction to augment in the input node feature space for better performance. We propose a simple but effective solution, **FLAG** (Free Large-scale Adversarial Augmentation on Graphs), which iteratively augments node features with gradient-based adversarial perturbations during training, and boosts performance at test time. Empirically, FLAG can be easily implemented with a dozen lines of code and is flexible enough to function with any GNN backbone, on a wide variety of large-scale datasets, and in both transductive and inductive settings. Without modifying a model's architecture or training setup, FLAG yields a consistent and salient performance boost across both node and graph classification tasks. Using FLAG, we reach state-of-the-art performance on the large-scale `ogbg-molpcba`, `ogbg-ppa`, and `ogbg-code` datasets.

## Experiments

To reproduce experimental results for **DeeperGCN**, visit [here](https://github.com/devnkong/FLAG/tree/main/deep_gcns_torch/examples/ogb).

Other baselines including **GCN**, **GraphSAGE**, **GAT**, **GIN**, **MLP**, etc. are available [here](https://github.com/devnkong/FLAG/tree/main/ogb).

To view the empirical performance of FLAG, please visit the Open Graph Benchmark [Node](https://ogb.stanford.edu/docs/leader_nodeprop/) an [Graph](https://ogb.stanford.edu/docs/leader_graphprop/) classification leaderboards.

## Requirements
  - ogb=1.2.3
  - torch-geometric=1.6.1
  - torch=1.5.0

## Citing FLAG

If you use FLAG in your work, please cite our paper.

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
