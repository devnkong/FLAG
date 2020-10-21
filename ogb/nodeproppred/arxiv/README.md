# ogbn-arxiv

To train baselines with FLAG in the default setup, run

**MLP+FLAG**
                    
        python mlp.py

**GCN+FLAG**

        python gnn.py

**GraphSAGE+FLAG**
        
        python gnn.py --use_sage

**GAT+FLAG**
        
        python gat_dgl/gat.py --use-norm --use-labels
