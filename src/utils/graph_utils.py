import dgl
import time
import torch
import random
import pickle
import numpy as np

import networkx as nx
import scipy.sparse as ssp

from tqdm import tqdm


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''
    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for idx, adj in enumerate(adj_list):
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
        # dats += [idx + 1, ] * len(adjcoo.data.tolist())
        
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def get_paths(source, target, adj, max_path_len=1, max_nodes_per_hop=None):                                         
    paths_dic = {}
    # ensure the max_path_len >= 2
    assert max_path_len >= 2
    # extract paths for len 1 ~ (max_len - 1)
    for i in range(1, max_path_len):
        paths_dic[i] = []
    for path_len in paths_dic.keys():
        # print("Path len: {}".format(path_len))
        if path_len == 1:
            paths = bfs_paths(adj, source, source, max_nodes_per_hop)
            paths_dic[path_len] = np.expand_dims(paths, 1).tolist()
        else:
            if len(paths_dic[path_len - 1]) == 0:
                break
            for path in paths_dic[path_len - 1]:
                root_node = path[path_len - 2][1]
                last_node = path[path_len - 2][0]
                paths = bfs_paths(adj, root_node, last_node, max_nodes_per_hop)
                if len(paths) > 0:
                    last_paths = np.repeat(np.expand_dims(path, 0), len(paths), 0)
                    paths = np.expand_dims(paths, 1)
                    paths = np.concatenate((last_paths, paths), axis=1)
                    paths_dic[path_len] += paths.tolist()
    # extract paths for len max_len
    last_paths = np.array(paths_dic[max_path_len - 1])
    if len(last_paths) > 0:
        paths_dic[max_path_len] = bfs_paths_for_last_hop(last_paths, target, max_path_len, adj)
    # only remain the paths from the source to the target node
    for path_len, paths in paths_dic.items():
        remained_paths = []
        for path in paths:
            if path[0][0] == source and path[path_len - 1][1] == target:
                remained_paths.append(path)
        paths_dic[path_len] = remained_paths
                
    return paths_dic


def bfs_paths(adj, node, visited, max_nodes_per_hop=None):
    neighbor_nodes = set(adj[node].nonzero()[1])
    neighbor_nodes = np.array(list(neighbor_nodes - set([visited])))
    if len(neighbor_nodes) > max_nodes_per_hop:
        neighbor_nodes = np.random.choice(neighbor_nodes, max_nodes_per_hop, replace=False)
    if len(neighbor_nodes) > 0:
        source_nodes = np.array([node]).repeat(len(neighbor_nodes))
        paths = np.array((source_nodes, neighbor_nodes)).transpose()
        return paths
    else:
        return []


def bfs_paths_for_last_hop(last_paths, target_node, max_path_len, adj_list):
    target_node_adj = np.array(adj_list.todense()[target_node])[0]
    last_nodes = last_paths[:, max_path_len - 2, 1]
    visited_nodes = last_paths[:, max_path_len - 2, 0]
    visited_idx = np.where(visited_nodes == target_node)[0]
    path_exists = target_node_adj[last_nodes]
    path_exists[visited_idx] = 0
    path_exists_idx = np.nonzero(path_exists)[0]
    selected_last_paths = np.array(last_paths[path_exists_idx])
    last_hops = np.concatenate((np.expand_dims(last_nodes[path_exists_idx], 1), np.expand_dims(np.array([target_node, ] * len(path_exists_idx)), 1)), 1)
    
    return np.concatenate((selected_last_paths, np.expand_dims(last_hops, 1)), 1).tolist()


def prepare_node_labels(source, target, paths_dic):
    node_labels = {}
    node_labels[source] = np.array([0, 1])
    node_labels[target] = np.array([1, 0])
    
    for path_len, paths in paths_dic.items():
        for path in paths:
            for i in range(0, path_len):
                h, t = path[i]
                h_label = [i, path_len - i]
                t_label = [i + 1, path_len - i - 1]
                # store h label
                if h in node_labels:
                    node_labels[h] = np.min(np.array([h_label, node_labels[h]]), axis=0)
                else:
                    node_labels[h] = h_label
                # store t label
                if t in node_labels:
                    node_labels[t] = np.min(np.array([t_label, node_labels[t]]), axis=0)
                else:
                    node_labels[t] = t_label
    return node_labels


def prepare_subgraph(paths_dic, node_labels, A_list, source, target, r_label, max_path_len):
    edges = []
    for path_len, paths in paths_dic.items():
        for path in paths:
            for i in range(0, path_len):
                edges.append(path[i])
    edges = np.array(edges)
    # create subgraph according to edges
    if len(edges) > 0:
        # remove the repeated edges
        src, dst = edges.transpose()
        subgraph_Adj = ssp.csc_matrix((np.ones(len(src)), (src, dst)))
        subgraph_Adjcoo = subgraph_Adj.tocoo()
        src, dst = subgraph_Adjcoo.row, subgraph_Adjcoo.col
        # get the tripletss according to src and dst
        triplets = []
        for i, Adj in enumerate(A_list):
            data = np.array(Adj[src, dst]).squeeze()
            nonzero_index = np.nonzero(data)[0]
            if len(nonzero_index) > 0:
                heads = src[nonzero_index]
                tails = dst[nonzero_index]
                rels = [i, ] * len(heads)
                triplets += np.array((heads, rels, tails)).transpose().tolist()
        triplets = np.array(triplets)
        # prepare the subgraph
        src, rel, dst = triplets.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        subgraph = dgl.graph((src, dst))
        subgraph.edata['type'] = torch.LongTensor(rel)
        subgraph.edata['label'] = torch.LongTensor(np.ones(subgraph.edata['type'].shape) * r_label)
    else:
        uniq_v, edges = np.unique(([source], [target]), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        subgraph = dgl.graph((src, dst))
        subgraph.remove_edges(0)
        subgraph.edata['type'] = torch.LongTensor([])
        subgraph.edata['label'] = torch.LongTensor([])
    node_labels = np.array([node_labels[v] for v in uniq_v])
    node_features = node_labels_to_features(node_labels, max_path_len)
    subgraph.ndata['feat'] = torch.FloatTensor(node_features)
    # label the source node and target node, the source node is labeled as 1, the target node is labeled as 2
    s_labels = np.zeros_like(uniq_v)
    source_idx = np.where(uniq_v == source)[0][0]
    s_labels[source_idx] = 1
    subgraph.ndata['s_label'] = torch.tensor(s_labels)
    t_labels = np.zeros_like(uniq_v)
    target_idx = np.where(uniq_v == target)[0][0]
    t_labels[target_idx] = 1
    subgraph.ndata['t_label'] = torch.tensor(t_labels)
    
    return subgraph


def node_labels_to_features(node_labels, max_len):
    num_nodes = len(node_labels)
    node_features = np.zeros((num_nodes, max_len * 2))
    node_features[np.arange(num_nodes), node_labels[:, 0]] = 1
    node_features[np.arange(num_nodes), max_len + node_labels[:, 1]] = 1
    return node_features


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000):
    pos_edges = edges
    neg_edges = []
    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]
    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)
    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        # random sample head/tail entity
        if np.random.uniform() < 0.5:
            neg_head = np.random.choice(n)
        else:
            neg_tail = np.random.choice(n)
        # ensure the head entity is not equal to the tail entity
        # the sampled triplets is not golden triplets
        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)
    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl
