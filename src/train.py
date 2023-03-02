import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning

from utils.initialize_utils import initialize_experiment, initialize_model
from data_process.datasets import generate_subgraph_datasets, SubgraphDataset
from models.graph_classifier import GraphClassifier as dgl_model
from managers.evaluator import Evaluator
from managers.trainer import Trainer

# from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
# from utils.initialization_utils import initialize_experiment, initialize_model
# from utils.graph_utils import collate_dgl, move_batch_to_device_dgl

from warnings import simplefilter


def main(params):
    # simplefilter(action='ignore', category=UserWarning)
    # simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/subgraphs_neg_{params.num_neg_samples_per_link}_len_{params.max_path_len}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths, params.triplets_form,
                            add_transpose_rels=params.add_transpose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            file_name=params.train_file)
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths, params.triplets_form,
                            add_transpose_rels=params.add_transpose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            file_name=params.valid_file)

    if params.add_transpose_rels:
        params.num_rels = train.num_rels * 2
    params.inp_dim = params.max_path_len * 2

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = params.max_path_len

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    logging.info(f"Device: {params.device}")
    if params.add_transpose_rels:
        logging.info(f"Input dim : {params.inp_dim}, # Relations : {int(params.num_rels / 2)}, # Augmented relations : {params.num_rels}")
    else:
        logging.info(f"Input dim : {params.inp_dim}, # Relations : {int(params.num_rels / 2)}")

    valid_evaluator = Evaluator(params, graph_classifier, valid)

    trainer = Trainer(params, graph_classifier, train, valid_evaluator)

    logging.info('Starting training with full batch...')

    trainer.train()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="base_train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")
    
    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=100,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=2,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=2,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=25,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--triplets_form", type=str, default='htr',
                        help="The form of the input triplets")
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--max_path_len", type=int, default=3,
                        help="max path len from source to target")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=100,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--add_transpose_rels', '-tr', type=bool, default=True,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", '-nw', type=int, default=8,
                        help="Number of dataloading processes")
    
    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--tlf_emb_dim", "-tlfd", type=int, default=32,
                        help="tlf embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--tlf_dropout", type=float, default=0.5,
                        help="Dropout rate in TLF module")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument('--has_normalization', '-nm', type=bool, default=True,
                        help="whether use normalization")
    parser.add_argument('--sefm_decoder', '-sd', type=str, default='DistMult', choices=['DistMult', 'TransE', 'FCN'],
                        help="choose whether decoder to use in SEFM")
                 

    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.triples'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/add_0/{}.triples'.format(params.dataset, params.valid_file))
    }

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    main(params)
