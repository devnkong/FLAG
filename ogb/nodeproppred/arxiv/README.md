# ogbn-arxiv

To train baselines with FLAG in the default setup, run

**MLP+FLAG**

The baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/arxiv).
                    
    python mlp.py

**GCN+FLAG**

The baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/arxiv).

    python gnn.py

**GraphSAGE+FLAG**

The baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/arxiv).
        
    python gnn.py --use_sage

**GAT+FLAG**

The baseline model [here](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv). We are using the `GAT(norm. adj.)+labels` version.
        
    python gat_dgl/gat.py --use-norm --use-labels
