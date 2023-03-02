import os
import sys
import time
import json
import torch
import logging


def initialize_experiment(params, file_name):
    """create the checkpoint dir
       create logfile
       store params.json
    Args:
        params (argparse): the parameters for this experiment
        file_name (str): the filename of "train" or "test"
    """
    # confirm main dir and add system path
    params.main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.join(params.main_dir, 'src'))
    
    # create experiments dir (if not exist)
    exps_dir = os.path.join(params.main_dir, 'experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    # create this experiment
    # just debug when the experiment is not set
    if params.experiment_name != 'default':
        params.experiment_name = "{}_{}".format(params.experiment_name, time.strftime('%Y%m%d%H%M%S'))
    params.exp_dir = os.path.join(exps_dir, params.experiment_name)
    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    # create log file
    if 'test' in file_name:
        params.test_exp_dir = os.path.join(params.exp_dir, f"test_{params.dataset}_{params.experiment_name}")
        if not os.path.exists(params.test_exp_dir):
            os.makedirs(params.test_exp_dir)
        file_handler = logging.FileHandler(os.path.join(params.test_exp_dir, f"test.log"))
    else:
        file_handler = logging.FileHandler(os.path.join(params.exp_dir, "train.log"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params, model, load_model=False):
    '''
    relation2id: the relation to id mapping, this is stored in the model and used when testing
    model: the type of model to initialize/load
    load_model: flag which decide to initialize the model or load a saved model
    '''

    if load_model and os.path.exists(os.path.join(params.exp_dir, 'best_graph_classifier.pth')):
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_graph_classifier.pth'))
        graph_classifier = torch.load(os.path.join(params.exp_dir, 'best_graph_classifier.pth')).to(device=params.device)
    else:
        relation2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
        with open(relation2id_path) as f:
            relation2id = json.load(f)

        logging.info('No existing model found. Initializing new model..')
        graph_classifier = model(params, relation2id).to(device=params.device)

    return graph_classifier