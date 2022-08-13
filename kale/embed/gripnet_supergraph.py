import torch

class SuperVertex(object):
	def __init__(self, name: str, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor=None, edge_weight: torch.Tensor=None) -> None:
		r"""
		The supervertex structure in GripNet. Each supervertex is a subgraph containing nodes with the same category and at least keep semantically-coherent. Supervertices can be homogeneous or heterogeneous.

		Args:
			name (str): the name of the supervertex.
			node_feat (torch.Tensor): node features of the supervertex with shape [#nodes, #features]. We recommend using `torch.sparse.FloatTensor()` if the node feature matrix is sparse.
			edge_index (torch.Tensor): edge indices in COO format with shape [2, #edges].
			edge_type (torch.Tensor, optional): one-dimensional relation type for each edge, indexed from 0. Defaults to None.
			edge_weight (torch.Tensor, optional): one-dimensional weight for each edge. Defaults to None.
		"""
		self.name = name
		self.node_feat = node_feat
		self.edge_index = edge_index
		self.edge_type = edge_type
		self.edge_weight = edge_weight

		# get the number of nodes, node features and edges
		self.n_node, self.n_node_feat = node_feat.shape
		self.n_edge = edge_index.shape[1]

		# initialize in-supervertex and out-supervertex lists
		self.in_supervertex_list = []
		self.out_supervertex_list = []
		self.if_start_supervertex = True

		self.__process_edges__()

	def __process_edges__(self):
		r"""
		process the edges of the supervertex.
		"""
		# get the number of edge types
		if self.edge_type is None:
			self.n_edge_type = 1
		else:
			unique_edge_type = self.edge_type.unique()
			self.n_edge_type = unique_edge_type.shape[0]
		
			# check if the index of edge type is continuous and starts from 0
			assert self.n_edge_type == unique_edge_type.max() + 1, 'The index of edge type is not continuous and starts from 0.'

	def __repr__(self) -> str:
		return f'SuperVertex(\n name={self.name}, \n  node_feat={self.node_feat.shape}, \n  edge_index={self.edge_index.shape}, \n  n_edge_type={self.n_edge_type})'

	def add_in_supervertex(self, vertex_name: str):
		self.in_supervertex_list.append(vertex_name)

	def add_out_supervertex(self, vertex_name: str):
		self.out_supervertex_list.append(vertex_name)

	def set_name(self, vertex_name: str):
		self.name = vertex_name