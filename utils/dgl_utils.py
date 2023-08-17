import dgl
import torch


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    graphs_pos, r_labels_pos, graphs_negs, r_labels_negs = map(list, zip(*samples))
    
    batched_graph_pos = dgl.batch(graphs_pos)
    r_labels_pos = torch.LongTensor(r_labels_pos)
    g_labels_pos = torch.ones_like(r_labels_pos)

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]
    
    batched_graph_neg = dgl.batch(graphs_neg)
    r_labels_neg = torch.LongTensor(r_labels_neg)
    g_labels_neg = torch.zeros_like(r_labels_neg)
    
    return (batched_graph_pos, r_labels_pos), g_labels_pos, (batched_graph_neg, r_labels_neg), g_labels_neg

def collate_dgl_CoMPILE(samples):
    # The input `samples` is a list of pairs
    graphs_pos, r_labels_pos, graphs_negs, r_labels_negs = map(list, zip(*samples))
    
    r_labels_pos = torch.LongTensor(r_labels_pos)
    g_labels_pos = torch.ones_like(r_labels_pos)

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    r_labels_neg = torch.LongTensor([item for sublist in r_labels_negs for item in sublist])
    g_labels_neg = torch.zeros_like(r_labels_neg)
    
    return (graphs_pos, r_labels_pos), g_labels_pos, (graphs_neg, r_labels_neg), g_labels_neg


def move_batch_to_device_dgl(batch, device):
    ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg) = batch

    targets_pos = targets_pos.to(device=device)
    r_labels_pos = r_labels_pos.to(device=device)

    targets_neg = targets_neg.to(device=device)
    r_labels_neg = r_labels_neg.to(device=device)

    g_dgl_pos = g_dgl_pos.to(device)
    g_dgl_neg = g_dgl_neg.to(device)

    return ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg)
