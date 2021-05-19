import os
import numpy as np
import argparse
import tensorflow as tf
c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config = c)
from scipy.io import loadmat

from utils import *
from src import *


parser = argparse.ArgumentParser(description = 'CLI args for training')
parser.add_argument('--config', type = str,
                    default = '/home/sarthak/als_aai/cfg.yaml')
parser.add_argument('--gpu', type = int, default = 0,
                    help = 'assign gpu id')
cfg = parser.parse_args()


def main(config, gpu):
    """ Main module to read data, pre-process and begin training of AAI models based on user configuration.

    Parameters
    ----------
    config: main.Configuration
        configuration file 
    gpu: int
        gpu to be used for training 
    """

    assert isinstance(config, object)
    cfg = Configuration(config)

    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        #print(device_lib.list_local_devices())

    for fold in range(cfg.folds):
        print(f"Fold: {fold+1}")
        """
            If x-vectors are required, extract train and validation data w/ their corresponding x-vectors. 
            Based on subject conditions, begin training of AAI models. 
        """
        if cfg.x_vectors:
            x_train, y_train, e_train, x_val, y_val, e_val = data_fetch(cfg,fold, cfg.sub_conds) #fetch train and val data (dysarthric corpus)
            x_train, y_train, x_val, y_val = pad(x_train ,y_train, x_val, y_val) # pad variable length sequences
            trainer(x_train, y_train, x_val, y_val, fold, cfg, e_train, e_val) # init model and begin training based on user config
        else:
            x_train, y_train, x_val, y_val = data_fetch(cfg, fold, cfg.sub_conds)
            x_train, y_train, x_val, y_val = pad(x_train, y_train, x_val, y_val)
            trainer(x_train, y_train, x_val, y_val, fold, cfg, e_train = None, e_val = None)


if __name__  == '__main__':
    main(cfg.config, cfg.gpu)
