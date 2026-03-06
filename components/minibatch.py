import random
import numpy as np
from dgl import DGLGraph
import networkx as nx
from tqdm import tqdm

seed = 123
random.seed(seed)
np.random.seed(seed)

class NodeMinibatchIterator(object):

    def __init__(self, g, num_layers, batch_size,
                    nodes_list, feats, feats_t,
                    labels, concat):
        self.g = g
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.nodes_list = nodes_list
        self.feats = feats
        self.feats_t = feats_t
        self.labels = labels
        self.concat = concat

    def get_batchs_nid(self):
        random.shuffle(self.nodes_list)
        batchs_nid = []
        for i in range(int(len(self.nodes_list)/self.batch_size)):
            batchs_nid.append(self.nodes_list[i*self.batch_size:(i+1)*self.batch_size])
        if len(self.nodes_list) % self.batch_size != 0:
            batchs_nid.append(self.nodes_list[-(len(self.nodes_list)%self.batch_size):])
        return batchs_nid

    def get_mask(self, total_index, labeled_index):
        labeled_index = set(labeled_index)
        mask = []
        for i in range(len(total_index)):
            if total_index[i] in labeled_index:
                mask.append(i)
        return np.array(mask)
        
    def get_subgraphs(self):
        batchs_nid = self.get_batchs_nid()
        subgraphs = []
        subfeats = []
        subfeats_t = []
        sublabels = []
        submasks = []
        for batch_nid in tqdm(batchs_nid):
            # get subgraph with unlabeled nodes
            subgraph_nodes = set()
            for n_id in batch_nid:
                subgraph_node = set()
                subgraph_node.add(n_id)
                for i in range(self.num_layers):
                    for node in subgraph_node:
                        subgraph_node = subgraph_node | set(self.g.neighbors(node))
                subgraph_nodes = subgraph_nodes | subgraph_node
            subgraph = self.g.subgraph(subgraph_nodes)
            # get labels & feats — index by sorted node IDs to align with precomputed feats_t
            sorted_nodes = sorted(subgraph)
            sublabels.append(self.labels[sorted_nodes])
            subfeats.append(self.feats[sorted_nodes])
            subfeats_t.append(self.feats_t[sorted_nodes])
            # get mask
            submasks.append(self.get_mask(sorted_nodes, batch_nid))
            subgraph = DGLGraph(nx.relabel.convert_node_labels_to_integers(subgraph, ordering='sorted'))
            if not self.concat:
                subgraph.add_edges(subgraph.nodes(), subgraph.nodes())
            subgraphs.append(subgraph)

        return subgraphs, subfeats, subfeats_t, sublabels, submasks
