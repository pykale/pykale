import torch
from torch_geometric.data import Batch, Data


# Helper: Create a mock molecular graph
def create_mock_graph(num_nodes, num_edges, in_feats):
    x = torch.randn(num_nodes, in_feats)  # node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # random edges
    return Data(x=x, edge_index=edge_index)


def create_mock_batch_graph(batch_size, num_nodes=10, num_edges=20, in_feats=100):
    graphs = [create_mock_graph(num_nodes, num_edges, in_feats) for _ in range(batch_size)]
    return Batch.from_data_list(graphs)
