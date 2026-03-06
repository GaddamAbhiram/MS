import networkx as nx
import numpy as np
import pygsp
import random
import pandas as pd
from pydoc import locate
import argparse

class WaveletMachine:

    def __init__(self, G, node_id, sample_number):
     
        self.approximation = 100
        self.step_size = 20
        self.heat_coefficient = 1000.0
        self.node_label_type = str
        self.index = list(G.nodes())
        self.G = pygsp.graphs.Graph(nx.adjacency_matrix(G))
        self.number_of_nodes = G.number_of_nodes()
        self.node_id = node_id
        self.sample_number = sample_number
        self.steps = [x*self.step_size for x in range(self.sample_number)]

    def approximate_wavelet_calculator(self):
   
        self.real_and_imaginary = []
        for node in range(0,self.number_of_nodes):
            impulse = np.zeros((self.number_of_nodes))
            if node in self.node_id:
                impulse[node] = 1
                wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.G, self.chebyshev, impulse)
                self.real_and_imaginary.append([np.mean(np.exp(wavelet_coefficients*1*step*1j)) for step in self.steps])
            else:
                self.real_and_imaginary.append([0. for step in self.steps])
        self.real_and_imaginary = np.array(self.real_and_imaginary)


    def approximate_structural_wavelet_embedding(self):
      
        self.G.estimate_lmax()
        self.heat_filter = pygsp.filters.Heat(self.G, tau=[self.heat_coefficient])
        self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter, m=self.approximation)
        self.approximate_wavelet_calculator()

    def transform_and_save_embedding(self):
      
        self.approximate_structural_wavelet_embedding()
        self.real_and_imaginary = np.concatenate([self.real_and_imaginary.real, self.real_and_imaginary.imag], axis=1)
        columns_1 = ["reals_" + str(x) for x in range(self.sample_number)]
        columns_2 = ["imags_" + str(x) for x in range(self.sample_number)]
        columns = columns_1 + columns_2
        self.real_and_imaginary = pd.DataFrame(self.real_and_imaginary, columns=columns)
        self.real_and_imaginary.index = self.index
    
        if self.node_label_type in ("int", "int64") or self.node_label_type is int:
            self.real_and_imaginary.index = self.real_and_imaginary.index.astype(np.int64)
        else:
            self.real_and_imaginary.index = self.real_and_imaginary.index.astype(str)
        self.real_and_imaginary = self.real_and_imaginary.sort_index()

        return self.real_and_imaginary.values

def get_coding_feats(G, node_id, sample_number):
    node_id = set(node_id)
    total_node_id = sorted(G)
    G = nx.relabel.convert_node_labels_to_integers(G, ordering='sorted')
    new_node_id = []
    for i in range(len(total_node_id)):
        if total_node_id[i] in node_id:
            new_node_id.append(i)
    machine = WaveletMachine(G, new_node_id, sample_number)
    return machine.transform_and_save_embedding()
