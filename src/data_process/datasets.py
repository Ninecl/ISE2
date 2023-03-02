import os
import lmdb
import json
import logging

import numpy as np

from torch.utils.data import Dataset

from utils.data_utils import process_files, deserialize
from utils.graph_utils import sample_neg
from utils.subgraph_utils import links2subgraphs
from utils.graph_utils import ssp_multigraph_to_dgl


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):

    testing = 'test' in splits
    adj_list, tlf_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, params.triplets_form, saved_relation2id)
    
    if params.add_transpose_rels:
        adj_list_T = [adj.T for adj in adj_list]
        adj_list += adj_list_T

    data_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link, max_size=split['max_size'])
    '''
    if testing:
        directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)
    '''

    links2subgraphs(adj_list, graphs, params)


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, triplets_form='htr', included_relations=None, add_transpose_rels=False, num_neg_samples_per_link=1, file_name=''):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name

        ssp_graph, tlf_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(raw_data_paths, triplets_form, included_relations)
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_transpose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.tlf_list = tlf_list

        with self.main_env.begin() as txn:
            self.max_path_len = int.from_bytes(txn.get('max_path_len'.encode()), byteorder='little')

        logging.info(f"Max path length: {self.max_path_len}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)
        

    def __getitem__(self, index):
        # positive
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            link_pos, r_label_pos, g_label_pos, subgraph_pos = deserialize(txn.get(str_id))
            ht_tlf_pos = [self.tlf_list[link_pos[0]], self.tlf_list[link_pos[1]]]
        # negative
        # links_neg = []
        subgraphs_neg = []
        r_labels_neg = [] 
        g_labels_neg = []
        ht_tlfs_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                link_neg, r_label_neg, g_label_neg, subgraph_neg = deserialize(txn.get(str_id))
                ht_tlf_neg = [self.tlf_list[link_neg[0]], self.tlf_list[link_neg[1]]]
                # links_neg.append(link_neg)
                subgraphs_neg.append(subgraph_neg)
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)
                ht_tlfs_neg.append(ht_tlf_neg)

        return ht_tlf_pos, subgraph_pos, g_label_pos, r_label_pos, \
               ht_tlfs_neg, subgraphs_neg, g_labels_neg, r_labels_neg
               

    def __len__(self):
        return self.num_graphs_pos
