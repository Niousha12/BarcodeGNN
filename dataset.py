import pandas as pd

from dataset_utils import create_graph_from_sequence, create_graph_with_embeddings
from tqdm import tqdm
import pickle


class DNADataset:
    def __init__(self, file_path, k_mer=4, stride=1, data_count=None, truncate=None, task='order_name'):

        # task can be 'order_name', 'genus_name', 'family_name', 'species_name'

        self.k_mer = k_mer

        train_csv = pd.read_csv(file_path)
        self.barcodes = train_csv['nucleotides'].to_list()

        if data_count is not None:
            self.barcodes = self.barcodes[:data_count]
            if truncate is not None:
                self.barcodes = [seq[:truncate] for seq in self.barcodes]

        self.labels = train_csv[task].to_list()

        unique_labels = sorted(list(set(self.labels)))
        self.number_of_classes = len(unique_labels)

        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}

        with open(f'kmers_embedding/kmer_vectors_{self.k_mer}.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)

        self.graphs = [create_graph_from_sequence(seq, k=self.k_mer, label=self.label2idx[self.labels[i]],
                                                  kmer_embeddings=self.embeddings, stride=stride) for i, seq in
                       tqdm(enumerate(self.barcodes))]
        # self.graphs = [create_graph_from_sequence(seq, k=self.k_mer, label=self.label2idx[self.labels[i]],
        #                                           kmer_embeddings=None, stride=stride) for i, seq in
        #                tqdm(enumerate(self.barcodes))]

        node_count = []
        edge_count = []
        for graph in self.graphs:
            node_count.append(graph.num_nodes)
            edge_count.append(graph.edge_index.size(1))
        self.average_node_count = {sum(node_count) / len(node_count)}
        self.average_edge_count = {sum(edge_count) / len(edge_count)}

if __name__ == "__main__":
    temp = DNADataset('data/supervised_train.csv', k_mer=5, stride=5, task='genus_name')
    temp2 = DNADataset('data/supervised_test.csv', k_mer=5, stride=5, task='order_name')
    print(temp.average_node_count)
    print(temp.average_edge_count)
    print(temp.number_of_classes)