import argparse
import datetime
import os
import time

import sklearn
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import optim
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch import nn

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

    train_dataset = DNADataset('data/supervised_train.csv', k_mer=args['kmer'], stride=stride, task=args['task'])
    test_dataset = DNADataset('data/unseen.csv', k_mer=args['kmer'], stride=stride, task=args['task'])

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
    model.classifier = nn.Identity()
    model = model.to(device)

    model.load_state_dict(model_weights)

    # DataLoader to handle mini-batching
    train_loader = DataLoader(train_dataset.graphs, batch_size=args['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset.graphs, batch_size=args['batch_size'], shuffle=False)

    model.eval()
    embeddings = []
    labels = []
    for data in train_loader:
        with torch.no_grad():
            emb = model(data.to(device))
            embeddings.append(emb)
            labels.append(data.label)

    print("Generating embeddings for train set", flush=True)
    X = torch.vstack(embeddings)  # expected shape (#data, embedding_dim)
    y = torch.stack(labels)  # expected shape #data

    embeddings = []
    labels = []
    for data in test_loader:
        with torch.no_grad():
            emb = model(data.to(device))
            embeddings.append(emb)
            labels.append(data.label)

    print("Generating embeddings for test set", flush=True)
    X_unseen = torch.vstack(embeddings)  # expected shape (#data, embedding_dim)
    y_unseen = torch.stack(labels)  # expected shape #data

    c = 0
    for label in y_unseen:
        if label not in y:
            c += 1
    print(f"There are {c} genus that are not present during training.")

    # kNN =====================================================================
    print("Computing Nearest Neighbors")

    # Fit ---------------------------------------------------------------------
    clf = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    clf.fit(X, y)

    # Evaluate ----------------------------------------------------------------
    # Create results dictionary
    results = {}
    for partition_name, X_part, y_part in [("Train", X, y), ("Unseen", X_unseen, y_unseen)]:
        y_pred = clf.predict(X_part)
        res_part = {}
        res_part["count"] = len(y_part)
        # Note that these evaluation metrics have all been converted to percentages
        res_part["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_part, y_pred)
        res_part["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_part, y_pred)
        res_part["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="micro")
        res_part["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="macro")
        res_part["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="weighted")
        results[partition_name] = res_part
        print(f"\n{partition_name} evaluation results:")
        for k, v in res_part.items():
            if k == "count":
                print(f"  {k + ' ':.<21s}{v:7d}")
            else:
                print(f"  {k + ' ':.<24s} {v:6.2f} %")


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
