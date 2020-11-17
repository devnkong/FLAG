import argparse

import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import numpy as np

import time
import sys
sys.path.insert(0,'../..')
from attacks import *

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def train(model, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def train_flag(model, x, y_true, train_idx, optimizer, args, device) :

    forward = lambda perturb : model(x[train_idx] + perturb)
    model_forward = (model, forward)
    y = y_true.squeeze(1)[train_idx]
    loss, _ = flag(model_forward, x[train_idx].shape, y, args, optimizer, device, F.nll_loss)

    return loss.item()

@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator):
    model.eval()

    out = model(x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)


    parser.add_argument('--step-size', type=float, default=2e-2)
    parser.add_argument('-m', type=int, default=3)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--attack', type=str, default='flag')
    parser.add_argument('--amp', type=float, default=2)
    parser.add_argument('--id', type=str, default='test')


    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    x = data.x
    if args.use_node_embedding:
        embedding = torch.load('embedding.pt', map_location='cpu')
        x = torch.cat([x, embedding], dim=-1)
    x = x.to(device)

    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    model = MLP(x.size(-1), args.hidden_channels, dataset.num_classes, args.num_layers,
                args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-products')

    vals, tests = [], []
    for run in range(args.runs):
        best_val, final_test = 0, 0

        seed = run + args.start_seed
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        start = time.time()

        for epoch in range(1, args.epochs + 1):
            # loss = train(model, x, y_true, train_idx, optimizer)
            loss = train_flag(model, x, y_true, train_idx, optimizer, args, device)
            if epoch > 0 and epoch % args.test_freq == 0 or epoch == args.epochs:
                result = test(model, x, y_true, split_idx, evaluator)
                train, val, tst = result
                if val > best_val:
                    best_val = val
                    final_test = tst

        print(f'Run{run} val:{best_val}, test:{final_test}')
        vals.append(best_val)
        tests.append(final_test)

    print('')
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")



if __name__ == "__main__":
    main()
