	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/1.pth > logs/1.log 2>&1
	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/2.pth > logs/2.log 2>&1
	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/3.pth > logs/3.log 2>&1
	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/4.pth > logs/4.log 2>&1
	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/5.pth > logs/5.log 2>&1
	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/6.pth > logs/6.log 2>&1
	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/7.pth > logs/7.log 2>&1
	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/8.pth > logs/8.log 2>&1
	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/9.pth > logs/9.log 2>&1
	python -u test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --data-folder () --model_load_path checkpoints/10.pth > logs/10.log 2>&1


    
    
# ogbn-products
We simply apply a random partition to generate batches for mini-batch training on GPU and full-batch test on CPU. We set the number of partitions to be 10 for training and the batch size is 1 subgraph.
## Default 
	--use_gpu False 
	--self_loop False
	--cluster_number 10
    --block res+ 	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax_sg, softmax, power]
    --num_layers 3
	--mlp_layers 1
    --norm batch
    --hidden_channels 128
    --epochs 500
    --lr 0.01
	--dropout 0.5
## ResGEN
### Train
	python main.py --use_gpu --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1
	python main.py --use_gpu --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 --same-epoch --unsym --data-folder () --device ()

### Test (use pre-trained model, [download](https://drive.google.com/file/d/1OxyA2IZN-4BCfkWzUG8QBS-khxhHHnZB/view?usp=sharing) from Google Drive)
	python test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1
