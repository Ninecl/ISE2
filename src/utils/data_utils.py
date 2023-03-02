import os
import pdb
import pickle
import logging

import dgl
import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.sparse import csc_matrix


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    [n1, n2], r_label, g_label, subgraph = data_tuple
    return [n1, n2], r_label, g_label, subgraph


def read_triplets(file_path, mode='hrt'):
    """read triplets from file according to mode
    return triplets in the term of '(h, t, r)'

    Args:
        file_path (str): file path
        mode (str, optional): input triplets form. Defaults to 'hrt'.

    Returns:
        list: the list of triplets in the form of (h, t, r)
    """
    f = open(file_path, 'r')
    file_data = [line.split() for line in f.readlines()]
    triplets = []
    for line in file_data:
        if mode == 'hrt':
            triplets.append([line[0], line[2], line[1]])
        elif mode == 'htr':
            triplets.append([line[0], line[1], line[2]])
        else:
            raise Exception("The form of input triplets is wrong.")
    return triplets


def process_files(files, triplets_form, saved_relation2id=None):
    """Read the data from files

    Args:
        files (str): file_path
        saved_relation2id (dict, optional): the saved relation2id dic. Defaults to None.

    Returns:
        _type_: _description_
    """
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets_dic = {}
    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            triplets = read_triplets(file_path, triplets_form)

        for triplet in triplets:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[1] not in entity2id:
                entity2id[triplet[1]] = ent
                ent += 1
            if not saved_relation2id and triplet[2] not in relation2id:
                relation2id[triplet[2]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[2] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])

        triplets_dic[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to each relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets_dic['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets_dic['train'][:, 0][idx].squeeze(1), triplets_dic['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # 构建type-level feature
    num_entity = len(entity2id)
    num_relation = len(relation2id)
    tlf_list = np.zeros((num_entity, num_relation*2))
    for h, t, r in triplets_dic['train']:
        tlf_list[h][r] += 1
        tlf_list[t][r+num_relation] += 1
        
    return adj_list, tlf_list, triplets_dic, entity2id, relation2id, id2entity, id2relation


def collate_dgl(samples):

    # The input `samples` is a list of pairs
    ht_tlfs_pos, graphs_pos, g_labels_pos, r_labels_pos, \
    ht_tlfs_negs, graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*samples))

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]
    # links_neg = [item for sublist in links_neg for item in sublist]
    ht_tlfs_neg = [item for sublist in ht_tlfs_negs for item in sublist]

    batched_graph_pos = dgl.batch(graphs_pos)
    batched_graph_neg = dgl.batch(graphs_neg)

    return (np.array(ht_tlfs_pos), batched_graph_pos, np.array(g_labels_pos), np.array(r_labels_pos)), \
           (np.array(ht_tlfs_neg), batched_graph_neg, np.array(g_labels_neg), np.array(r_labels_neg))


def move_batch_to_device_dgl(batch, device):
    (ht_tlfs_pos, graph_pos, g_labels_pos, r_labels_pos), \
    (ht_tlfs_neg, graph_neg, g_labels_neg, r_labels_neg) = batch

    # move tensor to device
    # links_pos = torch.LongTensor(links_pos).to(device=device)
    # links_neg = torch.LongTensor(links_neg).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)
    g_labels_pos = torch.LongTensor(g_labels_pos).to(device=device)
    g_labels_neg = torch.LongTensor(g_labels_neg).to(device=device)
    ht_tlfs_pos = torch.LongTensor(ht_tlfs_pos).to(device=device)
    ht_tlfs_neg = torch.LongTensor(ht_tlfs_neg).to(device=device)
    # move graph to device
    graph_pos = graph_pos.to(device=device)
    graph_neg = graph_neg.to(device=device)

    return (ht_tlfs_pos, graph_pos, r_labels_pos), g_labels_pos, \
           (ht_tlfs_neg, graph_neg, r_labels_neg), g_labels_neg


def replace_head(head, rel, tail, num_sample, n, adj_list):
    neg_samples = [[head, tail, rel], ]
    while len(neg_samples) < num_sample:
        neg_head = np.random.choice(n)
        if neg_head != tail and adj_list[rel][neg_head, tail] == 0:
            neg_samples.append([neg_head, tail, rel])
    return np.array(neg_samples)


def replace_tail(head, rel, tail, num_sample, n, adj_list):
    neg_samples = [[head, tail, rel], ]
    while len(neg_samples) < num_sample:
        neg_tail = np.random.choice(n)
        if neg_tail != head and adj_list[rel][head, neg_tail] == 0:
            neg_samples.append([head, neg_tail, rel])
    return np.array(neg_samples)


def replace_rel(head, rel, tail, num_sample, r, adj_list):
    neg_samples = [[head, tail, rel], ]
    while len(neg_samples) < num_sample:
        neg_rel = np.random.choice(r)
        if adj_list[neg_rel][head, tail] == 0:
            neg_samples.append([head, tail, neg_rel])
    return neg_samples


def get_neg_samples(test_links, adj_list, sample_mode, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]
    num_samples_ent = num_samples if num_samples < n else n
    num_samples_rel = num_samples if num_samples < r else r

    neg_triplets_dics = {}
    for (head, tail, rel) in tqdm(zip(heads, tails, rels), total=len(test_links)):
        # hrt sample
        if sample_mode == 'hrt':
            neg_triplet = {'h': [], 't': [], 'r': []}
            neg_triplet['h'] = replace_head(head, rel, tail, num_samples_ent, n, adj_list)
            neg_triplet['t'] = replace_tail(head, rel, tail, num_samples_ent, n, adj_list)
            neg_triplet['r'] = replace_rel(head, rel, tail, num_samples_rel, r, adj_list)
        # ht sample
        elif sample_mode == 'ht':
            neg_triplet = {'h': [], 't': []}
            neg_triplet['h'] = replace_head(head, rel, tail, num_samples_ent, n, adj_list)
            neg_triplet['t'] = replace_tail(head, rel, tail, num_samples_ent, n, adj_list)
        # h sample
        elif sample_mode == 'h':
            neg_triplet = {'h': []}
            neg_triplet['h'] = replace_head(head, rel, tail, num_samples_ent, n, adj_list)
        # t sample
        elif sample_mode == 't':
            neg_triplet = {'t': []}
            neg_triplet['t'] = replace_tail(head, rel, tail, num_samples_ent, n, adj_list)
        # r sample
        elif sample_mode == 'r':
            neg_triplet = {'r': []}
            neg_triplet['r'] = replace_rel(head, rel, tail, num_samples_rel, r, adj_list)
        
        neg_triplets_dics[f'{head} {tail} {rel}'] = neg_triplet

    return neg_triplets_dics


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets
