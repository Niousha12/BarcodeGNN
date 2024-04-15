import argparse
import datetime
import os

import torch
from torch import optim
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Linear
from tqdm import tqdm
from models import ShallowGCN, DeepGCN

from dataset import DNADataset


def main(args):
    if args['overlapping']:
        stride = 1
    else:
        stride = args['kmer']

    train_dataset = DNADataset('data/supervised_train.csv', k_mer=args['kmer'], stride=stride, task=args['task'])
    test_dataset = DNADataset('data/supervised_test.csv', k_mer=args['kmer'], stride=stride, task=args['task'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args['model'] == 'DeepGCN':
        model = DeepGCN(100, 64, train_dataset.number_of_classes)
    elif args['model'] == 'ShallowGCN':
        model = ShallowGCN(100, 64, train_dataset.number_of_classes)
    else:
        raise ValueError("Model not recognized.")

    model = model.to(device)

    # DataLoader to handle mini-batching
    train_loader = DataLoader(train_dataset.graphs, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset.graphs, batch_size=args['batch_size'], shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    # Training the GCN (A simple loop for demonstration; you should use proper training practices)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, total_iters=5)
    best_acc = 0
    acc_list = []
    loss_list = []

    saving_path = f"outputs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

    # Ensure saving path exists
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            out = model(data)

            loss = criterion(out, data.label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss {total_loss}")
        model.eval()
        correct = 0
        total = 0
        for data in test_loader:
            pred = model(data.to(device))
            correct += pred.argmax(dim=1).eq(data.label).sum().item()
            total += data.num_graphs
        acc = correct / total
        print(f"Accuracy: {acc}")

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}: lr {before_lr} -> {after_lr}")

        if acc > best_acc:
            best_acc = acc

            checkpoint = {
                'args': args,
                'model': model.state_dict(),
                'loss_list': loss_list,
                'acc_list': acc_list,
                'epoch': epoch}

            torch.save(checkpoint, os.path.join(saving_path, f"best_{args['model']}_{args['kmer']}_{args['task']}.pth"))


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', type=str, default='DeepGCN')
    parser.add_argument('--kmer', action='store', type=int, default=3)
    parser.add_argument('--overlapping', action='store', type=bool, default=True)
    parser.add_argument('--task', action='store', type=str, default='species_name')
    parser.add_argument('--batch_size', action='store', type=int, default=32)
    parser.add_argument('--lr', action='store', type=float, default=1e-4)
    parser.add_argument('--weight_decay', action='store', type=float, default=1e-05)
    parser.add_argument('--epochs', type=int, default=80, help='Total number of training epochs')

    args_parsed = parser.parse_args()
    args_dict = vars(args_parsed)

    for task in ['species_name', 'genus_name', 'order_name']:
        args_dict['task'] = task

        for overlapping in [True, False]:
            args_dict['overlapping'] = overlapping

            for kmer_size in [3, 4, 5]:
                args_dict['kmer'] = kmer_size

                for network in ['DeepGCN', 'ShallowGCN']:
                    args_dict['model'] = network

                    args_log = "\n".join("{}\t{}".format(k, v) for k, v in sorted(args_dict.items(), key=lambda t: str(t[0])))

                    print(args_log)
                    main(args_dict)
