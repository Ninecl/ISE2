import os
import json
import time
import pickle
import random
import logging
import argparse

import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import dgl

from warnings import simplefilter
from scipy.sparse import SparseEfficiencyWarning

from utils.data_utils import get_neg_samples
from utils.graph_utils import incidence_matrix
from utils.subgraph_utils import get_paths, prepare_node_labels, prepare_subgraph


simplefilter(action='ignore', category=SparseEfficiencyWarning)


def read_triplets(files, batch):
    support_triplets = []
    test_triplets = []
    
    for i in range(0, batch + 1):
        for file_type, file_path in files[i].items():
            with open(file_path) as f:
                file_data = [line.split() for line in f.read().split('\n')[:-1]]
                
                if file_type == 'support':
                    support_triplets += file_data
                
                if file_type == 'query' and i == batch:
                    test_triplets += file_data
    
    return support_triplets, test_triplets

    
def process_triplets(support_triplets, query_triplets, saved_relation2id, add_traspose_rels, mode='htr'):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    # relation2id is static
    relation2id = saved_relation2id
    # entity2id will envolving
    entity2id = {}

    triplets = {'support': support_triplets, 'query': query_triplets}
    ent = len(entity2id)
    rel = len(relation2id)

    for file_type, facts in triplets.items():
        data = []
        for triplet in facts:
            if mode == 'htr':
                h, t, r = triplet
            elif mode == 'hrt':
                h, r, t = triplet
            else:
                raise Exception('The form of triplets is wrong.')
            
            if h not in entity2id:
                entity2id[h] = ent
                ent += 1
            if t not in entity2id:
                entity2id[t] = ent
                ent += 1

            # Save the triplets corresponding to only the known relations
            if r in saved_relation2id:
                data.append([entity2id[h], entity2id[t], saved_relation2id[r]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['support'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['support'][:, 0][idx].squeeze(1), triplets['support'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))
    # Add transpose matrices to handle both directions of relations.
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list += adj_list_t
        
    # 构建type-level feature
    num_entity = len(entity2id)
    num_relation = len(relation2id)
    tlf_list = np.zeros((num_entity, num_relation*2))
    for h, t, r in triplets['support']:
        tlf_list[h][r] += 1
        tlf_list[t][r+num_relation] += 1
        
    return adj_list, np.array(tlf_list), triplets, entity2id, relation2id, id2entity, id2relation


def save_to_txt_file(file_path, neg_triplets, id2entity, id2relation):
    with open(file_path, "w") as f:
        for triplets in neg_triplets:
            for k, v in triplets.items():
                for s, o, r in v:
                    f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def save_to_pickle_file(file_path, neg_triplets):

    with open(file_path, "wb") as f:
        pickle.dump(neg_triplets, f)


def intialize_worker(model, adj_list, params):
    global model_, adj_list_, params_
    model_, adj_list_, params_ = model, adj_list, params


def move_batch_data_to_device(graph_data, device):
    ht_tlfs, subgraphs, r_labels = graph_data
    # move tensor to device
    ht_tlfs = torch.LongTensor(ht_tlfs).to(device=device)
    r_labels = torch.LongTensor(r_labels).to(device=device)
    # move graph to device
    subgraphs = subgraphs.to(device=device)

    return (ht_tlfs, subgraphs, r_labels)


def get_subgraphs_from_dic(links, save_subgraphs_pkl_path, tlf_list):
    h, t, r = links[0]
    pkl_file_path = os.path.join(save_subgraphs_pkl_path, f'{h} {t} {r}.pickle')
    with open(pkl_file_path, 'rb') as f:
        link_subgraph_dic = pickle.load(f)
    datas = [link_subgraph_dic[f'{h} {t} {r}'] for h, t, r in links]
    links, subgraphs, r_labels = map(list, zip(*datas))
    links = np.array(links)
    h_tlfs = np.expand_dims(tlf_list[links[:, 0]], 1)
    t_tlfs = np.expand_dims(tlf_list[links[:, 1]], 1)
    ht_tlfs = np.concatenate((h_tlfs, t_tlfs), 1)
    subgraphs = dgl.batch(subgraphs)
    return (ht_tlfs, subgraphs, r_labels)


def get_rank(neg_links, save_subgraphs_pkl_path, tlf_list):
    tail_neg_links = neg_links['t']
    graph_data = get_subgraphs_from_dic(tail_neg_links, save_subgraphs_pkl_path, tlf_list)
    graph_data = move_batch_data_to_device(graph_data, params_.device)
    tail_scores = model_(graph_data)
    tail_scores = tail_scores.squeeze(1).detach().cpu().numpy()
    tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == 0) + 1

    return tail_scores, tail_rank


def subgraph_extraction_labeling(ind, rel, A_list, max_path_len=1, max_nodes_per_hop=None):
    try:
        source, target = ind
        A_incidence = incidence_matrix(A_list)
        # extract the (1, h)-len path from the source to target node
        paths = get_paths(source, target, A_incidence, max_path_len, max_nodes_per_hop)
        node_labels = prepare_node_labels(source, target, paths)
        subgraph = prepare_subgraph(paths, node_labels, A_list, source, target, rel, max_path_len)
    except:
        print(source, target, rel)
    
    return subgraph


def extract_subgraph(args_):
    (n1, n2, r_label) = args_
    subgraph = subgraph_extraction_labeling((n1, n2), r_label, adj_list_, params_.max_path_len, params_.max_nodes_per_hop)
    return (f'{n1} {n2} {r_label}', (n1, n2), subgraph, r_label)

    
def mp_subgraph_extraction(tripelts_dics, file_path):
    with mp.Pool(processes=None) as p:
        for link_key, triplets_dic in tqdm(tripelts_dics.items(), total=len(tripelts_dics)):
            all_triplets = []
            for k, v in triplets_dic.items():
                all_triplets.append(v)
            all_triplets = np.array([j for i in all_triplets for j in i])
            h, t, r = all_triplets.transpose()
            link_subgraph_dic = dict()
            args_ = zip(h, t, r)
            for (key, link, subgraph, r_label) in p.imap(extract_subgraph, args_):
                link_subgraph_dic[key] = (link, subgraph, r_label)
            save_path = os.path.join(file_path, f'{link_key}.pickle')
            save_to_pickle_file(save_path, link_subgraph_dic)
    
    return link_subgraph_dic


def save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')


def main(params):
    model = torch.load(params.model_path, map_location=params.device)
    
    # run test for each batch
    for emerging_batch_idx in range(1, params.emerging_batch_num + 1):
        logging.info(f"####################################")
        logging.info(f"### Test for the {emerging_batch_idx}st emerging KG ###")
        logging.info(f"####################################")
        # read triplets
        support_triplets, query_triplets = read_triplets(params.file_paths, emerging_batch_idx)
        adj_list, tlf_list, triplets, entity2id, relation2id, id2entity, id2relation = process_triplets(support_triplets, query_triplets, model.relation2id, params.add_traspose_rels)
        intialize_worker(model, adj_list, params)
        
        if params.sample_mode == 'h' or params.sample_mode == 't' or params.sample_mode == 'r':
            # set path
            save_triplets_txt_path = os.path.join('../data', params.dataset, f'add_{emerging_batch_idx}', f'sample_ranking_{params.sample_mode}.triples')
            save_triplets_pkl_path = os.path.join('../data', params.dataset, f'add_{emerging_batch_idx}', f'sample_ranking_{params.sample_mode}.pickle')
            save_subgraphs_pkl_path = os.path.join('../data', params.dataset, f'add_{emerging_batch_idx}', 'test_subgraphs_pkl')
            if not os.path.exists(save_subgraphs_pkl_path):
                os.makedirs(save_subgraphs_pkl_path)
            # load or sample data
            if os.path.exists(save_triplets_txt_path) and params.no_resample :
                with open(save_triplets_pkl_path, 'rb') as f:
                    neg_triplets_dics = pickle.load(f)
            else:
                logging.info(f"Sampling negative links for links in {emerging_batch_idx}st emerging KG")
                neg_triplets_dics = get_neg_samples(triplets['query'], adj_list, params.sample_mode, params.num_negative_sample)
                save_to_txt_file(save_triplets_txt_path, neg_triplets_dics.values(), id2entity, id2relation)
                save_to_pickle_file(save_triplets_pkl_path, neg_triplets_dics)
                # sample subgraphs
                logging.info(f"Extracting enclosing subgraphs for negative links in {emerging_batch_idx}st emerging KG")
                mp_subgraph_extraction(neg_triplets_dics, save_subgraphs_pkl_path)    

            # rank
            head_ranks = []
            tail_ranks = []
            rel_ranks = []
            all_ranks = []
            all_head_scores = []
            all_tail_scores = []
            
            with torch.no_grad():
                model_.eval()
                logging.info(f"Calculating ranks for links in {emerging_batch_idx}st emerging KG")
                for triplets in tqdm(list(neg_triplets_dics.values())):
                    tail_scores, tail_rank = get_rank(triplets, save_subgraphs_pkl_path, tlf_list)
                    # head_ranks.append(head_rank)
                    tail_ranks.append(tail_rank)
                    # rel_ranks.append(rel_rank)
                    
                    # all_head_scores += head_scores.tolist()
                    # all_tail_scores += tail_scores.tolist()

                # 统计head_rank
                # head_isHit1List = [x for x in head_ranks if x <= 1]
                # head_isHit5List = [x for x in head_ranks if x <= 5]
                # head_isHit10List = [x for x in head_ranks if x <= 10]
                # head_hits_1 = len(head_isHit1List) / len(head_ranks)
                # head_hits_5 = len(head_isHit5List) / len(head_ranks)
                # head_hits_10 = len(head_isHit10List) / len(head_ranks)
                # head_mrr = np.mean(1 / np.array(head_ranks))

                # 统计tail_rank
                tail_isHit1List = [x for x in tail_ranks if x <= 1]
                tail_isHit5List = [x for x in tail_ranks if x <= 5]
                tail_isHit10List = [x for x in tail_ranks if x <= 10]
                tail_hits_1 = len(tail_isHit1List) / len(tail_ranks)
                tail_hits_5 = len(tail_isHit5List) / len(tail_ranks)
                tail_hits_10 = len(tail_isHit10List) / len(tail_ranks)
                tail_mrr = np.mean(1 / np.array(tail_ranks))
                logger.info('TAIL RESULT: MRR | Hits@1 | Hits@5 | Hits@10 : {:.5f} | {:.5f} | {:.5f} | {:.5f}'.format(tail_mrr, tail_hits_1, tail_hits_5, tail_hits_10))

                # 统计rel_rank
                # rel_isHit1List = [x for x in rel_ranks if x <= 1]
                # rel_isHit5List = [x for x in rel_ranks if x <= 5]
                # rel_isHit10List = [x for x in rel_ranks if x <= 10]
                # rel_hits_1 = len(rel_isHit1List) / len(rel_ranks)
                # rel_hits_5 = len(rel_isHit5List) / len(rel_ranks)
                # rel_hits_10 = len(rel_isHit10List) / len(rel_ranks)
                # rel_mrr = np.mean(1 / np.array(rel_ranks))

                # 把三个列表拼起来就是all_rank
                # all_ranks = head_ranks + rel_ranks + tail_ranks
                # # 统计all_rank
                # all_isHit1List = [x for x in all_ranks if x <= 1]
                # all_isHit5List = [x for x in all_ranks if x <= 5]
                # all_isHit10List = [x for x in all_ranks if x <= 10]
                # all_hits_1 = len(all_isHit1List) / len(all_ranks)
                # all_hits_5 = len(all_isHit5List) / len(all_ranks)
                # all_hits_10 = len(all_isHit10List) / len(all_ranks)
                # all_mrr = np.mean(1 / np.array(all_ranks))

# return {'all_mrr': all_mrr, 'all_hits_1': all_hits_1, 'all_hits_5': all_hits_5, 'all_hits_10': all_hits_10, 
#         'head_mrr': head_mrr, 'head_hits_1': head_hits_1, 'head_hits_5': head_hits_5, 'head_hits_10': head_hits_10,
#         'tail_mrr': tail_mrr, 'tail_hits_1': tail_hits_1, 'tail_hits_5': tail_hits_5, 'tail_hits_10': tail_hits_10,
#         'rel_mrr': rel_mrr, 'rel_hits_1': rel_hits_1, 'rel_hits_5': rel_hits_5, 'rel_hits_10': rel_hits_10}

# save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)


# 写一个解析函数，方便点
def analyse_result(dic, s):
    return dic[f'{s}_mrr'], dic[f'{s}_hits_1'], dic[f'{s}_hits_5'], dic[f'{s}_hits_10']


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str,
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Path to dataset")
    parser.add_argument("--emerging_batch_num", "-ebn", type=int, default=5,
                        help="The number of emerging batch")
    parser.add_argument("--sample_mode", "-m", type=str, default="sample", choices=["h", "t", "r", 'ht', 'hrt'],
                        help="Negative sampling mode")
    parser.add_argument('--max_path_len', '-mpl', type=int, default=3,
                        help='The max path length')
    parser.add_argument("--max_nodes_per_hop", type=int, default=100,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=True,
                        help='Whether to append adj matrix list with symmetric relations?')
    parser.add_argument('--use_cuda', '-uc', type=bool, default=True,
                        help='Whether use cuda.')
    parser.add_argument('--device', '-de', type=int, default=0, choices=[-1, 0, 1, 2, 3],
                        help='Which gpu to use.')
    parser.add_argument('--num_negative_sample', '-ns', type=int, default=100,
                        help='Number of negative sample for each link.')
    parser.add_argument('--no_resample', '-nrs', action='store_true',
                        help='Whether resample negative links.')
    parser.add_argument('--model_name', '-mn', type=str, default='best_graph_classifier.pth',
                        help='Which model to use.')
    parser.add_argument('--test_times', '-tt', type=int, default=1, 
                        help='How many times to test.')

    params = parser.parse_args()

    file_handler = logging.FileHandler(os.path.join('../experiments', params.experiment_name, f'rank_test_{time.time()}.log'))

    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')
    
    # the file paths
    params.file_paths = {
        0: {
            'support': os.path.join('../data', params.dataset, 'base_train.triples'),
            'query': os.path.join('../data', params.dataset, 'add_0', 'valid.triples')
        },
        1:{
            'support': os.path.join('../data', params.dataset, 'add_1', 'pruned_support.triples'),
            'query': os.path.join('../data', params.dataset, 'add_1', 'test.triples')
        },
        2:{
            'support': os.path.join('../data', params.dataset, 'add_2', 'pruned_support.triples'),
            'query': os.path.join('../data', params.dataset, 'add_2', 'test.triples')
        },
        3:{
            'support': os.path.join('../data', params.dataset, 'add_3', 'pruned_support.triples'),
            'query': os.path.join('../data', params.dataset, 'add_3', 'test.triples')
        },
        4:{
            'support': os.path.join('../data', params.dataset, 'add_4', 'pruned_support.triples'),
            'query': os.path.join('../data', params.dataset, 'add_4', 'test.triples')
        },
        5:{
            'support': os.path.join('../data', params.dataset, 'add_5', 'pruned_support.triples'),
            'query': os.path.join('../data', params.dataset, 'add_5', 'test.triples')
        }
    }
    params.model_path = os.path.join('../experiments', params.experiment_name, params.model_name)
    
    # set device
    if params.use_cuda and torch.cuda.is_available() and params.device >= 0:
        params.device = torch.device('cuda:%d' % params.device)
    else:
        params.device = torch.device('cpu')

    sum_all_mrr = []
    sum_all_hits_1 = []
    sum_all_hits_5 = []
    sum_all_hits_10 = []

    for i in range(params.test_times):
        result = main(params)
        # for s in ['tail']:
        #     mrr, hits_1, hits_5, hits_10 = analyse_result(result, s)
        #     if s == 'all':
        #         sum_all_mrr.append(mrr)
        #         sum_all_hits_1.append(hits_1)
        #         sum_all_hits_5.append(hits_5)
        #         sum_all_hits_10.append(hits_10)
        #     logger.info('{} RESULT: MRR | Hits@1 | Hits@5 | Hits@10 : {:.5f} | {:.5f} | {:.5f} | {:.5f}'.format(s.upper(), mrr, hits_1, hits_5, hits_10))

    # mean_mrr = np.mean(sum_all_mrr)
    # mean_hits_1 = np.mean(sum_all_hits_1)
    # mean_hits_5 = np.mean(sum_all_hits_5)
    # mean_hits_10 = np.mean(sum_all_hits_10)

    # logger.info('Test {} times.\nMean result: MRR | Hits@1 | Hits@5 | Hits@10 : {:.5f} | {:.5f} | {:.5f} | {:.5f}'.format(params.test_times, mean_mrr, mean_hits_1, mean_hits_5, mean_hits_10))