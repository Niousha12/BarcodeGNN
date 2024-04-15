import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import add_self_loops, degree, scatter
from torch.nn import Linear as Lin, Sequential as Seq
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, ResGraphBlock, DenseGraphBlock


class WeightedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(WeightedGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.linear(x)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Step 4: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # Message passing with edge weights.
        return norm.view(-1, 1) * x_j


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = WeightedGCNConv(3, 16)  # Assuming input features are 3-dimensional
        self.conv2 = WeightedGCNConv(16, 2)  # Assuming 2 classes for output

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GCN2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch_index):
        # First Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Second Graph Convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        x = scatter(x, batch_index, dim=0, reduce='mean')

        return x
        # return F.log_softmax(x, dim=1)


# Assume data object has .x for node features, .edge_index for edge indices, and .edge_attr for edge weights.


class ShallowGCN(torch.nn.Module):
    def __init__(self, num_features, embedding_size, num_classes):
        # Init parent
        super(ShallowGCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size * 2, num_classes)

    def forward(self, data):
        x, edge_index, batch_index = data.x, data.edge_index, data.batch
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out


class DeepGCN(torch.nn.Module):
    """
    static graph

    """

    def __init__(self, num_features, embedding_size, num_classes):
        super(DeepGCN, self).__init__()
        channels = embedding_size
        act = 'relu'
        norm = 'batch'
        bias = True
        conv = 'mr'
        heads = 1
        c_growth = 0
        self.n_blocks = 14
        self.head = GraphConv(num_features, channels, conv, act, norm, bias, heads)
        block = 'res'
        dropout = 0.2

        res_scale = 1 if block.lower() == 'res' else 0
        if block.lower() == 'dense':
            c_growth = channels
            self.backbone = MultiSeq(*[DenseGraphBlock(channels + i * c_growth, c_growth, conv, act, norm, bias, heads)
                                       for i in range(self.n_blocks - 1)])
        else:
            self.backbone = MultiSeq(*[ResGraphBlock(channels, conv, act, norm, bias, heads, res_scale)
                                       for _ in range(self.n_blocks - 1)])
        fusion_dims = int(channels * self.n_blocks + c_growth * ((1 + self.n_blocks - 1) * (self.n_blocks - 1) / 2))
        self.fusion_block = MLP([fusion_dims, 1024], act, None, bias)
        self.prediction = Seq(*[MLP([1 + fusion_dims, 512], act, norm, bias), torch.nn.Dropout(p=dropout),
                                MLP([512, 256], act, norm, bias), torch.nn.Dropout(p=dropout),
                                MLP([256, num_classes], None, None, bias)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        feats = [self.head(x, edge_index)]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1], edge_index)[0])
        feats = torch.cat(feats, 1)
        fusion, _ = torch.max(self.fusion_block(feats), 1, keepdim=True)
        out = self.prediction(torch.cat((feats, fusion), 1))

        out = scatter(out, batch, dim=0, reduce='mean')

        return out


if __name__ == "__main__":
    model = ShallowGCN(100, 64, 13)
    print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    model = DeepGCN(100, 64, 13)
    print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
