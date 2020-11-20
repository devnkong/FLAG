# FLAG

To run in the default setup

## ogbg-molhiv

	python main.py --use_gpu --conv_encode_edge --num_layers 7 --dataset ogbg-molhiv --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.2 --step-size 1e-2
	
## ogbg-molpcba

	python main.py --use_gpu --conv_encode_edge --add_virtual_node --mlp_layers 2 --num_layers 14 --dataset ogbg-molpcba --block res+ --gcn_aggr softmax_sg --t 0.1 --step-size 8e-3

