import argparse
import os
import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset

def load_graph_data(training_config, device):
    basepath = "D:/training_data/"
    filenames = []
    for root, dirs, files in os.walk(basepath):
        for name in files:
            name = os.path.join(basepath, name)
            filenames.append(name)

    # Collect train/val/test graphs here
    edge_index_list = []
    node_ap_features_list = []
    node_st_features_list = []
    node_labels_list = []

    # num_graphs_per_split_cumulative = [0]

    for datafile in filenames[::]:
        print("Loading dataset "+datafile)
        f = open(datafile)
        tracklets = json.load(f)
        random.shuffle(tracklets)
        for tracklet in tracklets[:]:
            node_ap_features_list.append(torch.Tensor(tracklet[0]))
            node_st_features_list.append(torch.Tensor(tracklet[1]))
            node_labels_list.append(torch.Tensor(tracklet[2]).to(torch.int64))
            edge_index_list.append(torch.Tensor(tracklet[3]).to(torch.int64))

    total_length = len(node_labels_list)
    train_split_index = int(0.5 * total_length)
    val_split_index = int(0.8 * total_length)
    num_graphs_per_split_cumulative = [0, train_split_index, val_split_index, total_length]

    # Optimization, do a shortcut in case we only need the test data loader
    if training_config['ppi_load_test_only']:
        data_loader_test = GraphDataLoader(
            node_ap_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            node_st_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            batch_size=training_config['batch_size'],
            shuffle=False
        )
        return data_loader_test
    else:

        data_loader_train = GraphDataLoader(
            node_ap_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            node_st_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
            batch_size=training_config['batch_size'],
            shuffle=True
        )

        data_loader_val = GraphDataLoader(
            node_ap_features_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
            node_st_features_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
            node_labels_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
            edge_index_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
            batch_size=training_config['batch_size'],
            shuffle=False  # no need to shuffle the validation and test graphs
        )

        data_loader_test = GraphDataLoader(
            node_ap_features_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
            node_st_features_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
            node_labels_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
            edge_index_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
            batch_size=training_config['batch_size'],
            shuffle=False
        )
        return data_loader_train, data_loader_val, data_loader_test


class GraphDataLoader(DataLoader):
    """
    When dealing with batches it's always a good idea to inherit from PyTorch's provided classes (Dataset/DataLoader).

    """
    def __init__(self, node_ap_features_list, node_st_features_list, node_labels_list, edge_index_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_ap_features_list, node_st_features_list, node_labels_list, edge_index_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


class GraphDataset(Dataset):
    """
    This one just fetches a single graph from the split when GraphDataLoader "asks" it

    """
    def __init__(self, node_ap_features_list, node_st_features_list, node_labels_list, edge_index_list):
        self.node_ap_features_list = node_ap_features_list
        self.node_st_features_list = node_st_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        return self.node_ap_features_list[idx], self.node_st_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx]


def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from PPI as defined by the batch size
    and merge them into a single graph with multiple connected components.

    It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise
    the scatter functions in the implementation 3 will fail.

    :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    """

    edge_index_list = []
    node_ap_features_list = []
    node_st_features_list = []
    node_labels_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_ap_features_list.append(features_labels_edge_index_tuple[0])
        node_st_features_list.append(features_labels_edge_index_tuple[1])
        node_labels_list.append(features_labels_edge_index_tuple[2])

        edge_index = features_labels_edge_index_tuple[3]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[2])  # update the number of nodes we've seen so far

    # Merge the PPI graphs into a single graph with multiple connected components
    node_ap_features = torch.cat(node_ap_features_list, 0)
    node_st_features = torch.cat(node_st_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    return node_ap_features, node_st_features, node_labels, edge_index
