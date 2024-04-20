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
from models import ShallowGCN, DeepGCN, GatGCN

from dataset import DNADataset


def evaluate(args, model_weights):
    if args['overlapping']:
        stride = 1
    else:
        stride = args['kmer']

    # train_dataset = DNADataset('data/supervised_train.csv', k_mer=args['kmer'], stride=stride, task=args['task'])
    test_dataset = DNADataset('data/supervised_test.csv', k_mer=args['kmer'], stride=stride, task=args['task'])

    # assert train_dataset.number_of_classes == test_dataset.number_of_classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args['model'] == 'DeepGCN':
        model = DeepGCN(100, 64, test_dataset.number_of_classes)
    elif args['model'] == 'ShallowGCN':
        model = ShallowGCN(100, 64, test_dataset.number_of_classes)
    elif args['model'] == 'GatGCN':
        model = GatGCN(100, 64, test_dataset.number_of_classes)
    else:
        raise ValueError("Model not recognized.")

    model = model.to(device)

    model.load_state_dict(model_weights)

    # DataLoader to handle mini-batching
    test_loader = DataLoader(test_dataset.graphs, batch_size=args['batch_size'], shuffle=False)

    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        with torch.no_grad():
            pred = model(data.to(device))
            correct += pred.argmax(dim=1).eq(data.label).sum().item()
            total += data.num_graphs
    acc = correct / total
    print(f"Accuracy: {acc}")


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', action='store', type=str,
                        default='outputs/2024-04-15_14-44-57/best_DeepGCN_3_species_name.pth')

    args_parsed = parser.parse_args()
    args_dict = vars(args_parsed)

    checkpoint = torch.load(args_dict['checkpoint_path'])

    print(checkpoint['args'])
    print('trained epochs:', checkpoint['epoch'])
    evaluate(checkpoint['args'], checkpoint['model'])

    # file_paths = []
    #
    # # os.walk generates the file names in a directory tree
    # for dir_path, _, filenames in os.walk('outputs'):
    #     for filename in filenames:
    #         # Create the relative path
    #         relative_path = os.path.relpath(os.path.join(dir_path, filename), start='outputs')
    #         # Append the relative path to the list
    #         file_paths.append(relative_path)
    #
    # for filename in sorted(file_paths):
    #     dir_path = 'outputs'
    #     args_dict['checkpoint_path'] = os.path.join(dir_path, filename)
    #     checkpoint = torch.load(args_dict['checkpoint_path'])
    #     print(checkpoint['args'])
    #     print('trained epochs:', checkpoint['epoch'])
    #     evaluate(checkpoint['args'], checkpoint['model'])

