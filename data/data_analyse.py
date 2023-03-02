import os
import random
import numpy as np


def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index


def read_entities(dataset, batch):
    path = os.path.join('{}'.format(dataset), 'add_{}'.format(batch), 'node.pgrk')
    with open(path, 'r') as f:
        nodes_set = set([i.split()[0] for i in f.readlines()])
    return nodes_set


def read_triplets(dataset, batch):
    if batch == 1:
        path = os.path.join('{}'.format(dataset), 'base_train.triples')
    else:
        path = os.path.join('{}'.format(dataset), 'add_{}'.format(batch), 'support.triples')
    with open(path, 'r') as f:
        triplets = [i.split() for i in f.readlines()]
    return triplets


def read_test_triplets(dataset, batch):
    path = os.path.join('{}'.format(dataset), 'add_{}'.format(batch), 'test.triples')
    with open(path, 'r') as f:
        triplets = [i.split() for i in f.readlines()]
    return triplets


def prune_test_triplets(triplets, entities):
    pruned_triplets = []
    for h, t, r in triplets:
        if h not in entities and t not in entities:
            pruned_triplets.append([h, t, r])
    return pruned_triplets


def split_unseen(triplets, seen_entities):
    unseen_entities = set()
    for h, t, r in triplets:
        if h not in seen_entities:
            unseen_entities.add(h)
        if t not in seen_entities:
            unseen_entities.add(t)
    return unseen_entities


def dependence_analyse(triplets, unseen_entities, batch_entities):
    u2b_triplets = []
    for h, t, r in triplets:
        if (h in unseen_entities and t in batch_entities) or (h in batch_entities and t in unseen_entities):
            u2b_triplets.append([h, t, r])
    return u2b_triplets


def unseen2unseen_analyse(triplets, unseen_entities):
    unseen_triplets = []
    for h, t, r in triplets:
        if h in unseen_entities and t in unseen_entities:
            unseen_triplets.append([h, t, r])
    return unseen_triplets


def write_triplets(triplets, dataset, batch_num, filename):
    with open(f"./{dataset}/add_{batch_num}/{filename}.triples", 'w') as f:
        for h, t, r in triplets:
            f.write(f"{h}\t{t}\t{r}\n")


if __name__ == "__main__":
    
    dataset = "WN-MBE"
    # 统计方式0: 只把original里的当作seen
    # 统计方式1: 只把每个batch的看作unseen，前一个batch之前的都看作seen，并统计各个batch之间的三元组数
    mode = 1
    # 是否对数据进行剪枝，剪枝策略为随机采样当前batch与original之间的数据条数，使其与u2u数量相等
    prune = 1
    
    # 读取entity2id和relation2id
    entity2id, id2entity = load_index(os.path.join(dataset, 'entity2id.txt'))
    relation2id, id2relation = load_index(os.path.join(dataset, 'relation2id.txt'))
    
    # 统计original KG中数据
    original_entities = read_entities(dataset, 1)
    original_triplets = read_triplets(dataset, 1)
    print("Original KG中共有{}个实体, {}条三元组".format(len(original_entities), len(original_triplets)))
    seen_entities = set([i[0] for i in original_triplets]).union(set([i[1] for i in original_triplets]))
    print("其中共有{}个已知实体".format(len(seen_entities)))
    print()
    
    # batch1
    batch1_entities = read_entities(dataset, 2)
    batch1_triplets = read_triplets(dataset, 2)
    print("Batch1 KG中共有{}个实体, {}条三元组".format(len(batch1_entities), len(batch1_triplets)))
    batch1_unseen_entities = split_unseen(batch1_triplets, seen_entities)
    print("其中共有{}个未知实体".format(len(batch1_unseen_entities)))
    batch1_u2u_triplets = unseen2unseen_analyse(batch1_triplets, batch1_unseen_entities)
    print("Batch1 KG中{}条三元组为unseen-to-unseen的链接".format(len(batch1_u2u_triplets)))
    # test里只保存prune后有的entity
    batch1_test_triplets = read_test_triplets(dataset, 2)
    batch1_test_u2u_triplets = prune_test_triplets(batch1_test_triplets, original_entities)
    write_triplets(batch1_test_u2u_triplets, dataset, 2, 'test_u2u')
    if mode == 1: 
        b1_to_o = dependence_analyse(batch1_triplets, batch1_unseen_entities, original_entities)
        print("Batch1 KG中{}条三元组为新实体与Original KG的链接".format(len(b1_to_o)))
        seen_entities = batch1_entities
    print()
    
    # batch2
    batch2_entities = read_entities(dataset, 3)
    batch2_triplets = read_triplets(dataset, 3)
    print("Batch2 KG中共有{}个实体, {}条三元组".format(len(batch2_entities), len(batch2_triplets)))
    batch2_unseen_entities = split_unseen(batch2_triplets, seen_entities)
    print("其中共有{}个未知实体".format(len(batch2_unseen_entities)))
    batch2_u2u_triplets = unseen2unseen_analyse(batch2_triplets, batch2_unseen_entities)
    print("{}条三元组为unseen-to-unseen的链接".format(len(batch2_u2u_triplets)))
    # test里只保存prune后有的entity
    batch2_test_triplets = read_test_triplets(dataset, 3)
    batch2_test_u2u_triplets = prune_test_triplets(batch2_test_triplets, original_entities)
    write_triplets(batch2_test_u2u_triplets, dataset, 3, 'test_u2u')
    if mode == 1:
        b2_to_o = dependence_analyse(batch2_triplets, batch2_unseen_entities, original_entities)
        b2_to_b1 = dependence_analyse(batch2_triplets, batch2_unseen_entities, batch1_unseen_entities)
        print("{}条三元组为新实体与Original KG的链接".format(len(b2_to_o)))
        print("{}条三元组为新实体与Batch1 KG的链接".format(len(b2_to_b1)))
        seen_entities = batch2_entities
    print()
    
    # batch3
    batch3_entities = read_entities(dataset, 4)
    batch3_triplets = read_triplets(dataset, 4)
    print("Batch3 KG中共有{}个实体, {}条三元组".format(len(batch3_entities), len(batch3_triplets)))
    batch3_unseen_entities = split_unseen(batch3_triplets, seen_entities)
    print("其中共有{}个未知实体".format(len(batch3_unseen_entities)))
    batch3_u2u_triplets = unseen2unseen_analyse(batch3_triplets, batch3_unseen_entities)
    print("{}条三元组为unseen-to-unseen的链接".format(len(batch3_u2u_triplets)))
    # test里只保存prune后有的entity
    batch3_test_triplets = read_test_triplets(dataset, 4)
    batch3_test_u2u_triplets = prune_test_triplets(batch3_test_triplets, original_entities)
    write_triplets(batch3_test_u2u_triplets, dataset, 4, 'test_u2u')
    if mode == 1:
        b3_to_o = dependence_analyse(batch3_triplets, batch3_unseen_entities, original_entities)
        b3_to_b1 = dependence_analyse(batch3_triplets, batch3_unseen_entities, batch1_unseen_entities)
        b3_to_b2 = dependence_analyse(batch3_triplets, batch3_unseen_entities, batch2_unseen_entities)
        print("{}条三元组为新实体与Original KG的链接".format(len(b3_to_o)))
        print("{}条三元组为新实体与Batch1 KG的链接".format(len(b3_to_b1)))
        print("{}条三元组为新实体与Batch2 KG的链接".format(len(b3_to_b2)))
        seen_entities = batch3_entities 
    print()
    
    # batch4
    batch4_entities = read_entities(dataset, 5)
    batch4_triplets = read_triplets(dataset, 5)
    print("Batch4 KG中共有{}个实体, {}条三元组".format(len(batch4_entities), len(batch4_triplets)))
    batch4_unseen_entities = split_unseen(batch4_triplets, seen_entities)
    print("其中共有{}个未知实体".format(len(batch4_unseen_entities)))
    batch4_u2u_triplets = unseen2unseen_analyse(batch4_triplets, batch4_unseen_entities)
    print("{}条三元组为unseen-to-unseen的链接".format(len(batch4_u2u_triplets)))
    # test里只保存prune后有的entity
    batch4_test_triplets = read_test_triplets(dataset, 5)
    batch4_test_u2u_triplets = prune_test_triplets(batch4_test_triplets, original_entities)
    write_triplets(batch4_test_u2u_triplets, dataset, 5, 'test_u2u')
    if mode == 1:
        b4_to_o = dependence_analyse(batch4_triplets, batch4_unseen_entities, original_entities)
        b4_to_b1 = dependence_analyse(batch4_triplets, batch4_unseen_entities, batch1_unseen_entities)
        b4_to_b2 = dependence_analyse(batch4_triplets, batch4_unseen_entities, batch2_unseen_entities)
        b4_to_b3 = dependence_analyse(batch4_triplets, batch4_unseen_entities, batch3_unseen_entities)
        print("{}条三元组为新实体与Original KG的链接".format(len(b4_to_o)))
        print("{}条三元组为新实体与Batch1 KG的链接".format(len(b4_to_b1)))
        print("{}条三元组为新实体与Batch2 KG的链接".format(len(b4_to_b2)))
        print("{}条三元组为新实体与Batch3 KG的链接".format(len(b4_to_b3)))
        seen_entities = batch4_entities
    print()
    
    # batch5
    batch5_entities = read_entities(dataset, 6)
    batch5_triplets = read_triplets(dataset, 6)
    print("Batch5 KG中共有{}个实体, {}条三元组".format(len(batch5_entities), len(batch5_triplets)))
    batch5_unseen_entities = split_unseen(batch5_triplets, seen_entities)
    print("其中共有{}个未知实体".format(len(batch5_unseen_entities)))
    batch5_u2u_triplets = unseen2unseen_analyse(batch5_triplets, batch5_unseen_entities)
    print("{}条三元组为unseen-to-unseen的链接".format(len(batch2_u2u_triplets)))
    # test里只保存prune后有的entity
    batch5_test_triplets = read_test_triplets(dataset, 6)
    batch5_test_u2u_triplets = prune_test_triplets(batch5_test_triplets, original_entities)
    write_triplets(batch5_test_u2u_triplets, dataset, 6, 'test_u2u')
    if mode == 1:
        b5_to_o = dependence_analyse(batch5_triplets, batch5_unseen_entities, original_entities)
        b5_to_b1 = dependence_analyse(batch5_triplets, batch5_unseen_entities, batch1_unseen_entities)
        b5_to_b2 = dependence_analyse(batch5_triplets, batch5_unseen_entities, batch2_unseen_entities)
        b5_to_b3 = dependence_analyse(batch5_triplets, batch5_unseen_entities, batch3_unseen_entities)
        b5_to_b4 = dependence_analyse(batch5_triplets, batch5_unseen_entities, batch4_unseen_entities)
        print("{}条三元组为新实体与Original KG的链接".format(len(b5_to_o)))
        print("{}条三元组为新实体与Batch1 KG的链接".format(len(b5_to_b1)))
        print("{}条三元组为新实体与Batch2 KG的链接".format(len(b5_to_b2)))
        print("{}条三元组为新实体与Batch3 KG的链接".format(len(b5_to_b3)))
        print("{}条三元组为新实体与Batch4 KG的链接".format(len(b5_to_b4)))
        seen_entities = batch5_entities
    print()
    
    
    # 裁剪数据
    if prune == 1:
        print("裁剪数据集: ")
        num_batches_triplets = len(b5_to_b1 + b5_to_b2 + b5_to_b3 + b5_to_b4 + batch5_u2u_triplets)
        num_pruned_triplets = int(num_batches_triplets / 5) + num_batches_triplets
        
        # 裁剪batch1
        num_pruned_b1_to_o = num_pruned_triplets - len(batch1_u2u_triplets)
        b1_to_o = random.sample(b1_to_o, min(len(b1_to_o), num_pruned_b1_to_o))
        b1_prune_triplets = b1_to_o + batch1_u2u_triplets
        print(f"Batch1保留{len(b1_prune_triplets)}条三元组, 其中与original KG链接的三元组保留{len(b1_to_o)}条")
        write_triplets(b1_prune_triplets, dataset, 2, 'pruned_support')
        # b1_no_triplets = batch1_u2u_triplets
        # write_triplets(b1_no_triplets, dataset, 2, 'no_support')
        # b1_depend_triplets = b1_to_o
        # write_triplets(b1_depend_triplets, dataset, 2, 'depend_support')
        
        # 裁剪batch2
        num_pruned_b2_to_o = num_pruned_triplets - len(batch2_u2u_triplets) - len(b2_to_b1)
        b2_to_o = random.sample(b2_to_o, min(len(b2_to_o), num_pruned_b2_to_o))
        b2_prune_triplets = b2_to_o + b2_to_b1 + batch2_u2u_triplets
        print(f"Batch2保留{len(b2_prune_triplets)}条三元组, 其中与original KG链接的三元组保留{len(b2_to_o)}条")
        write_triplets(b2_prune_triplets, dataset, 3, 'pruned_support')
        # b2_no_triplets = b2_to_b1 + batch2_u2u_triplets
        # write_triplets(b2_no_triplets, dataset, 3, 'no_support')
        # b2_depend_triplets = b2_to_b1
        # write_triplets(b2_depend_triplets, dataset, 3, 'depend_support')
        
        # 裁剪batch3
        num_pruned_b3_to_o = num_pruned_triplets - len(batch3_u2u_triplets) - len(b3_to_b2) - len(b3_to_b1)
        b3_to_o = random.sample(b3_to_o, min(len(b3_to_o), num_pruned_b3_to_o))
        b3_prune_triplets = b3_to_o + b3_to_b1 + b3_to_b2 + batch3_u2u_triplets
        print(f"Batch3保留{len(b3_prune_triplets)}条三元组, 其中与original KG链接的三元组保留{len(b3_to_o)}条")
        write_triplets(b3_prune_triplets, dataset, 4, 'pruned_support')
        # b3_no_triplets = b3_to_b1 + b3_to_b2 + batch3_u2u_triplets
        # write_triplets(b3_no_triplets, dataset, 4, 'no_support')
        # b3_depend_triplets = b3_to_b2
        # write_triplets(b3_depend_triplets, dataset, 4, 'depend_support')
        
        # 裁剪batch4
        num_pruned_b4_to_o = num_pruned_triplets - len(batch4_u2u_triplets) - len(b4_to_b3) - len(b4_to_b2) - len(b4_to_b1)
        b4_to_o = random.sample(b4_to_o, min(len(b4_to_o), num_pruned_b4_to_o))
        b4_prune_triplets = b4_to_o + b4_to_b1 + b4_to_b2 + b4_to_b3 + batch4_u2u_triplets
        print(f"Batch4保留{len(b4_prune_triplets)}条三元组, 其中与original KG链接的三元组保留{len(b4_to_o)}条")
        write_triplets(b4_prune_triplets, dataset, 5, 'pruned_support')
        # b4_no_triplets = b4_to_b1 + b4_to_b2 + b4_to_b3 + batch4_u2u_triplets
        # write_triplets(b4_no_triplets, dataset, 5, 'no_support')
        # b4_depend_triplets = b4_to_b3
        # write_triplets(b4_depend_triplets, dataset, 5, 'depend_support')
        
        # 裁剪batch5
        num_pruned_b5_to_o = num_pruned_triplets - len(batch5_u2u_triplets) - len(b5_to_b4) - len(b5_to_b3) - len(b5_to_b2) - len(b5_to_b1)
        b5_to_o = random.sample(b5_to_o, min(len(b5_to_o), num_pruned_b5_to_o))
        b5_prune_triplets = b5_to_o + b5_to_b1 + b5_to_b2 + b5_to_b3 + b5_to_b4 + batch5_u2u_triplets
        print(f"Batch5保留{len(b5_prune_triplets)}条三元组, 其中与original KG链接的三元组保留{len(b5_to_o)}条")
        write_triplets(b5_prune_triplets, dataset, 6, 'pruned_support')
        # b5_no_triplets = b5_to_b1 + b5_to_b2 + b5_to_b3 + b5_to_b4 + batch5_u2u_triplets
        # write_triplets(b5_no_triplets, dataset, 6, 'no_support')
        # b5_depend_triplets = b5_to_b4
        # write_triplets(b5_depend_triplets, dataset, 6, 'depend_support')