import os
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
from tdc.multi_pred import DTI
from utils import smile_to_graph, seq_cat
from tqdm import tqdm
import torch
import pickle


class DTIDataset(InMemoryDataset):
    def __init__(self, dataset, root, xd=None, xt=None, y=None, transform=None, pre_transform=None, smile_graph=None):
        # root is required for save preprocessed data, default is '/tmp'
        super(DTIDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: - list of targets (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        prot_seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        prot_seq_dict = {v: (i + 1) for i, v in enumerate(prot_seq_voc)}
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            target = seq_cat(target, prot_seq_dict)
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = "DAVIS"
    y_convert_log = True
    smile_graph_file = f"data/{dataset}_smile_graphs.pkl"
    data = DTI(name=dataset)
    if y_convert_log:
        data.convert_to_log()
    split = data.get_split()
    split_type = ['train', 'test', 'valid']
    load_smile_graph = True
    # build smile graphs
    if load_smile_graph:
        if not os.path.isfile(smile_graph_file):
            count = 0
            smile_graphs = {}
            for split_dataset in split_type:
                print(f"{dataset} {split_dataset} smile atomic graph constructing")
                for smile in tqdm(split[split_dataset]['Drug'].unique()):
                    if smile not in smile_graphs:
                        graph = smile_to_graph(smile)
                        if graph:
                            smile_graphs[smile] = graph
                            count += 1
            print(f"total {count} smiles are converted to mol graph")
            f = open(smile_graph_file, "wb")
            pickle.dump(smile_graphs, f)
            f.close()
        else:
            smile_graphs = pickle.load(open(smile_graph_file, "rb"))
            print(f"load {len(smile_graphs)} smile mol graphs")

    # build geometirc dataloader
    for split_dataset in split_type:
        drugs, proteins, y = list(split[split_dataset]['Drug']), list(split[split_dataset]['Target']), list(split[split_dataset]['Y'])
        DTIDataset(dataset=dataset + f"_{split_dataset}", root='data', xd=drugs, xt=proteins, y=y, smile_graph=smile_graphs)
