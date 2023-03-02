import os
import time
import random
import warnings

import torch
import numpy as np

import scipy.sparse as ssp

from test_rank import read_triplets, process_triplets
from utils.subgraph_utils import subgraph_extraction_labeling


# dataset = 'FB-MBE'
# model_path = os.path.join('../experiments/TSGCN_FB-MBE_20221207175823/best_graph_classifier.pth')
# device = torch.device('cuda:0')
# add_traspose_rels = True

# file_paths = {
#         0: {
#             'support': os.path.join('../data', dataset, 'base_train.triples'),
#             'query': os.path.join('../data', dataset, 'add_0', 'valid.triples')
#         },
#         1:{
#             'support': os.path.join('../data', dataset, 'add_1', 'pruned_support.triples'),
#             'query': os.path.join('../data', dataset, 'add_1', 'test.triples')
#         },
#         2:{
#             'support': os.path.join('../data', dataset, 'add_2', 'pruned_support.triples'),
#             'query': os.path.join('../data', dataset, 'add_2', 'test.triples')
#         },
#         3:{
#             'support': os.path.join('../data', dataset, 'add_3', 'pruned_support.triples'),
#             'query': os.path.join('../data', dataset, 'add_3', 'test.triples')
#         },
#         4:{
#             'support': os.path.join('../data', dataset, 'add_4', 'pruned_support.triples'),
#             'query': os.path.join('../data', dataset, 'add_4', 'test.triples')
#         },
#         5:{
#             'support': os.path.join('../data', dataset, 'add_5', 'pruned_support.triples'),
#             'query': os.path.join('../data', dataset, 'add_5', 'test.triples')
#         }
#     }
# model = torch.load(model_path, map_location=device)
# emerging_batch_idx = 1

# support_triplets, query_triplets = read_triplets(file_paths, emerging_batch_idx)
# adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_triplets(support_triplets, query_triplets, model.relation2id, add_traspose_rels)

# link = ['/m/0_ytw', '/m/09c7w0', '/base/biblioness/bibs_location/country']
# print(entity2id[link[0]], entity2id[link[1]], relation2id[link[2]])
    
row = [1, 0, 4, 3, 2, 4, 4]
col = [3, 4, 2, 4, 1, 0, 1]
data = [1, 2, 3, 4, 5, 6, 0]

matrix = ssp.csc_matrix((data, (row, col)))

print(matrix)
print()
print(matrix[4])
print()
print(matrix[4].nonzero()[1])