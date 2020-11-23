# ogbg-code

To train baselines with FLAG in the default setup, run

**GCN+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/code).

    python main_pyg.py --gnn gcn --step-size 8e-3

**GCN+V+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/code).
 
    python main_pyg.py --gnn gcn-virtual --step-size 8e-3

**GIN+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/code).

    python main_pyg.py --gnn gin --step-size 8e-3

**GIN+V+FLAG**, the baseline model [here](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/code).

    python main_pyg.py --gnn gin-virtual --step-size 8e-3

