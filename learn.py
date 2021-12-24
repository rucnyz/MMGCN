import torch
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, AGNNConv, GraphConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

x = torch.tensor([[1., 2], [2, 3], [1, 3]])
edge_index = torch.LongTensor([[0, 0], [1, 2]])
edge_index = to_undirected(edge_index)  # 处理成无向图
graph = Data(x = x, edge_index = edge_index)
gat = GATConv(2,1)
a = gat(x, edge_index)
a