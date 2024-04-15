import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def extract_kmers_from_sequence(sequence, k=3, stride=1):
    """Generate k-mers from a given DNA sequence."""
    kmers = []
    for i in range(0, len(sequence) - k + 1, stride):
        temp = sequence[i:i + k]
        if 'N' not in temp:
            kmers.append(temp)
    return kmers


def create_graph_from_sequence(sequence, k=3, label=0, kmer_embeddings=None, stride=1):
    """Convert a DNA sequence into a graph where k-mers are nodes."""
    kmers = extract_kmers_from_sequence(sequence, k, stride)
    kmer_indices = {kmer: i for i, kmer in enumerate(sorted(set(kmers)))}

    edges = []
    for i in range(len(kmers) - 1):
        source = kmer_indices[kmers[i]]
        target = kmer_indices[kmers[i + 1]]
        edges.append((source, target))

    if kmer_embeddings is not None:
        x = torch.tensor([kmer_embeddings[kmer] for kmer in sorted(set(kmer_indices.keys()))], dtype=torch.float)
    else:
        x = torch.randn(len(kmer_indices), 100)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # for kmer in kmer_indices:
    #     print(kmer, kmer_indices[kmer])

    # Creating a simple feature vector for each node (k-mer)
    # x = torch.randn(len(kmer_indices), 256)

    return Data(x=x, edge_index=edge_index, label=torch.tensor([label]), node_indices=list(kmer_indices.keys()))


def create_graph_with_embeddings(sequence, k=3, label=0, kmer_embeddings=None, stride=1, embedding_dim=100):
    """Create a graph from a sequence, initializing node features with k-mer embeddings, including edge weights."""
    kmers = extract_kmers_from_sequence(sequence, k, stride)
    features = []
    edges = []
    edge_weights = []
    kmer_to_node = {}

    for i, kmer in enumerate(kmers):
        if kmer not in kmer_to_node:
            kmer_to_node[kmer] = len(kmer_to_node)
            features.append(kmer_embeddings.get(kmer, [0.0] * embedding_dim))

        if i > 0:
            source = kmer_to_node[kmers[i - 1]]
            target = kmer_to_node[kmer]
            edges.append([source, target])
            edges.append([target, source])  # Assuming undirected graph

            # Simple weight calculation, here just a placeholder, use a real calculation based on your criteria
            weight = 1.0 / (abs(i - (i - 1)) + 1)  # Example: inverse of the position difference (simplified)
            edge_weights.append(weight)
            edge_weights.append(weight)  # Same weight for both directions

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    x = torch.tensor(features, dtype=torch.float)



    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, label=torch.tensor([label]))
