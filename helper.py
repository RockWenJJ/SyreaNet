import argparse
import yaml
import os
import logging
import json
import importlib
from datetime import datetime
from torch.utils.data import DataLoader

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/syreanet_test.yaml',
                        help='Path of the configuration file.')
    parser.add_argument('-l', '--load-path', type=str, default=None,
                        help='Path from which to load the network checkpoint.')
    parser.add_argument('-o', '--output-dir', type=str, default='./output',
                        help='Directory to save the results.')
    
    args = parser.parse_args()
    return args

def init_dataset(dataset_cfg):
    Datasets = importlib.import_module('dataset')
    try:
        Dataset = getattr(Datasets, dataset_cfg['type'])
    except Exception as e:
        raise NotImplementedError

    dataset = Dataset(**dataset_cfg['params'])
    
    return dataset

def init_dataloader(dataset_cfg):
    loaders = {"train":None, "valid":None, "test":None}
    for key in loaders.keys():
        if key in dataset_cfg.keys():
            set = init_dataset(dataset_cfg[key])
            loader_cfg = dataset_cfg[key]["loader"]
            loaders[key] = DataLoader(set, batch_size=loader_cfg['batch_size'], shuffle=True, 
                                      num_workers=loader_cfg['n_workers'])
    
    return loaders["train"], loaders["valid"], loaders["test"]


def init_network(network_cfg):
    Models = importlib.import_module('models')
    try:
        Model = getattr(Models, network_cfg['type'])
    except Exception as e:
        raise NotImplementedError
    
    net = Model(network_cfg)
    return net

def init_loss(loss_cfg):
    Losses = importlib.import_module('losses')
    try:
        Loss = getattr(Losses, loss_cfg['type'])
    except Exception as e:
        # print(e)
        raise NotImplementedError
    
    train_loss_func = Loss(loss_cfg["params"], mode="train")
    valid_loss_func = Loss(loss_cfg["params"], mode="valid")
    return train_loss_func, valid_loss_func


def init_logger(cfg):
    Loggers = importlib.import_module('logger')
    try:
        Logger = getattr(Loggers, cfg.logger["type"])
    except Exception as e:
        raise NotImplementedError
    logger = Logger(cfg)
    return logger

def init_preparer(cfg):
    import utils.prepares as Prep
    # train_prep = getattr(Prep, cfg['train']['type'])('train')
    # val_prep = getattr(Prep, cfg['valid']['type'])('valid')
    train_prep = None
    val_prep = None
    test_prep = getattr(Prep, cfg['test']['type'])('test')
    
    return train_prep, val_prep, test_prep