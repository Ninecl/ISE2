import lmdb
import time
import logging
import numpy as np
import multiprocessing as mp

from tqdm import tqdm

from utils.graph_utils import incidence_matrix, get_paths, prepare_node_labels, prepare_subgraph
from utils.data_utils import serialize


def links2subgraphs(A, graphs, params):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    intialize_worker(A, params)
    max_path_len = params.max_path_len
    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
    
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM

    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        if split_name == 'train':
            extraction_helper(A, split['pos'], labels, env, split_env, params, pos=True)
        else:
            extraction_helper(A, split['pos'], labels, env, split_env, params, pos=False)

        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, env, split_env, params, pos=False)

    with env.begin(write=True) as txn:
        bit_max_path_len = int.bit_length(int(max_path_len))
        txn.put('max_path_len'.encode(), (int(max_path_len)).to_bytes(bit_max_path_len, byteorder='little'))


def intialize_worker(A, params):
    global A_, params_
    A_, params_ = A, params


def extraction_helper(A, links, g_labels, env, split_env, params, pos):

    with env.begin(write=True, db=split_env) as txn:
        txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

    with mp.Pool(processes=15, initializer=intialize_worker, initargs=(A, params)) as p:
        args_ = zip(range(len(links)), links, g_labels)
        if pos:
            for (str_id, datum) in tqdm(p.imap(extract_pos_subgraph, args_), total=len(links)):
                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))
        else:
            for (str_id, datum) in tqdm(p.imap(extract_neg_subgraph, args_), total=len(links)):
                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))


def extract_pos_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    subgraph = pos_subgraph_extraction_labeling((n1, n2), r_label, A_, params_.max_path_len, params_.max_nodes_per_hop)
    datum = {'link': [n1, n2], 'r_label': r_label, 'g_label': g_label, 'subgraph': subgraph}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)


def extract_neg_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    subgraph = neg_subgraph_extraction_labeling((n1, n2), r_label, A_, params_.max_path_len, params_.max_nodes_per_hop)
    datum = {'link': [n1, n2], 'r_label': r_label, 'g_label': g_label, 'subgraph': subgraph}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        subgraph = neg_subgraph_extraction_labeling((n1, n2), r_label, A, params.max_path_len, params.max_nodes_per_hop)
        # ht_tlf = np.array([tlf_list[n1], tlf_list[n2]])
        datum = {'link': [n1, n2], 'r_label': r_label, 'g_label': 0, 'subgraph': subgraph}
        total_size += len(serialize(datum))

    return total_size / sample_size


def pos_subgraph_extraction_labeling(ind, rel, A_list, max_path_len=1, max_nodes_per_hop=None):
    source, target = ind
    A_list[rel][source, target] = 0 # This will change both A_list[rel] and A_list[rel + num_relation] in A_list
    A_incidence = incidence_matrix(A_list)
    # extract the (1, h)-len path from the source to target node
    paths = get_paths(source, target, A_incidence, max_path_len, max_nodes_per_hop)
    node_labels = prepare_node_labels(source, target, paths)
    subgraph = prepare_subgraph(paths, node_labels, A_list, source, target, rel, max_path_len)
    A_list[rel][source, target] = 1
    
    return subgraph


def neg_subgraph_extraction_labeling(ind, rel, A_list, max_path_len=1, max_nodes_per_hop=None):
    source, target = ind
    A_incidence = incidence_matrix(A_list)
    # extract the (1, h)-len path from the source to target node
    paths = get_paths(source, target, A_incidence, max_path_len, max_nodes_per_hop)
    node_labels = prepare_node_labels(source, target, paths)
    subgraph = prepare_subgraph(paths, node_labels, A_list, source, target, rel, max_path_len)
    
    return subgraph