"""
Graph Convolutional Network for Cell Spatial Graphs.

Architecture from Pathomic Fusion (Chen et al., 2022):
- 3 layers of SAGEConv + SAGPooling
- Global mean+max pooling at each layer
- Residual aggregation across layers
- FC head: nhid*2 → nhid → grph_dim → label_dim
- Input: Cell spatial graphs with 12 morphological + 1024 CPC features (1036 total)
- Graph construction: KNN with K=5 from nuclei segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn import SAGEConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from utils.initialization import init_max_weights


class NormalizeFeaturesV2(object):
    """Column-normalize the first 12 handcrafted features, keep CPC features as-is."""

    def __call__(self, data):
        if data.x.shape[1] > 12:
            max_vals = data.x[:, :12].max(0, keepdim=True)[0]
            max_vals[max_vals == 0] = 1  # Avoid division by zero
            data.x[:, :12] = data.x[:, :12] / max_vals
        else:
            max_vals = data.x.max(0, keepdim=True)[0]
            max_vals[max_vals == 0] = 1
            data.x = data.x / max_vals

        if torch.cuda.is_available():
            data.x = data.x.type(torch.cuda.FloatTensor)
        else:
            data.x = data.x.type(torch.FloatTensor)
        return data


class NormalizeEdgesV2(object):
    """Normalize edge attributes."""

    def __call__(self, data):
        if data.edge_attr is not None:
            if torch.cuda.is_available():
                data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
            else:
                data.edge_attr = data.edge_attr.type(torch.FloatTensor)
            max_vals = data.edge_attr.max(0, keepdim=True)[0]
            max_vals[max_vals == 0] = 1
            data.edge_attr = data.edge_attr / max_vals
        return data


class GraphNet(nn.Module):
    """
    Graph Convolutional Network for cell spatial graphs.
    
    Architecture:
    - 3 × (SAGEConv → ReLU → SAGPooling → global_mean+max_pool)
    - Residual aggregation: x = x1 + x2 + x3
    - FC classifier: nhid*2 → nhid → grph_dim → label_dim
    
    Args:
        features: Input feature dimension per node (default: 1036)
        nhid: Hidden dimension (default: 128)
        grph_dim: Output embedding dimension (default: 32)
        dropout_rate: Dropout probability (default: 0.25)
        GNN: GNN variant for SAGPooling (default: 'GCN')
        pooling_ratio: Fraction of nodes to keep per pooling (default: 0.20)
        act: Output activation function
        label_dim: Number of output classes
        init_max: Use SNN-style initialization
    """

    def __init__(self, features=1036, nhid=128, grph_dim=32,
                 nonlinearity=torch.tanh, dropout_rate=0.25,
                 GNN='GCN', use_edges=0, pooling_ratio=0.20,
                 act=None, label_dim=1, init_max=True):
        super(GraphNet, self).__init__()

        self.dropout_rate = dropout_rate
        self.use_edges = use_edges
        self.act = act

        # Graph convolution layers
        self.conv1 = SAGEConv(features, nhid)
        self.pool1 = SAGPooling(nhid, ratio=pooling_ratio, GNN=SAGEConv)
        self.conv2 = SAGEConv(nhid, nhid)
        self.pool2 = SAGPooling(nhid, ratio=pooling_ratio, GNN=SAGEConv)
        self.conv3 = SAGEConv(nhid, nhid)
        self.pool3 = SAGPooling(nhid, ratio=pooling_ratio, GNN=SAGEConv)

        # Classifier
        self.lin1 = nn.Linear(nhid * 2, nhid)
        self.lin2 = nn.Linear(nhid, grph_dim)
        self.lin3 = nn.Linear(grph_dim, label_dim)

        self.output_range = Parameter(
            torch.FloatTensor([6]), requires_grad=False
        )
        self.output_shift = Parameter(
            torch.FloatTensor([-3]), requires_grad=False
        )

        if init_max:
            init_max_weights(self)
            print("Initializing GraphNet with SNN-style weights")

    def forward(self, **kwargs):
        data = kwargs['x_grph']

        # Normalize features and edges
        data = NormalizeFeaturesV2()(data)
        data = NormalizeEdgesV2()(data)

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # Layer 1
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(
            x, edge_index, edge_attr, batch
        )
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Layer 2
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(
            x, edge_index, edge_attr, batch
        )
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Layer 3
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(
            x, edge_index, edge_attr, batch
        )
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Residual aggregation
        x = x1 + x2 + x3

        # Classifier
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        features = F.relu(self.lin2(x))
        out = self.lin3(features)

        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out
