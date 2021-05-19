import os
import numpy as np
from utils.Get_SPIRE_EMA_data_Full import Get_SPIRE_data


def ri_pad(x_train, y_train, x_val, y_val):
    """ Return the training and validation data of the dysarthric corpus, if RI AAI model is chosen.

    Parameters
    ----------
    x_train: list 
        Acoustic features of the dysarthric corpus, for training. 
    y_train: list 
        Articulatory features, for training. 
    x_val: list 
        Acoustic features, for model validation
    y_val: list 
        Articulatory features, for validation. 
    
    Returns
    -------
    Input parameters
    """
    return x_train, y_train, x_val, y_val


def mc_pad(x_train, y_train, x_val, y_val, cfg):
    """ Return the training and validation data of the dysarthric corpus + cross-corpus, if MC AAI model is chosen.
    
    Parameters
    ----------
    x_train: list 
        Acoustic features of the dysarthric corpus, for training. 
    y_train: list 
        Articulatory features, for training. 
    x_val: list 
        Acoustic features, for model validation
    y_val: list 
        Articulatory features, for validation. 
    cfg: main.Configuration
        Configuration file

    Returns
    -------
    x_train_new: list
        Cross-corpus + Dysarthric corpus training data
    y_train_1: list 
        Cross-corpus' articulatory trajectories by masking the dysarthric corpus's trajectories.
    y_train_2: list
        Dysarthric corpus' articulatory trajectories by masking the cross-corpus' trajectories.
    x_val_new: list
        Validation data from the dysarthric corpus only.
    y_val_1: list
        8-dim trajectories of the dysarthric corpus by masking the other 8-dims. 
    y_val_2: list
        8-dim trajectories of the dysarthric corpus by masking the other 8-dims. 
    """

    x_t, y_t, x_v, y_v = Get_SPIRE_data(cfg) # obtain data from the cross-corpus
    x_train_new = np.concatenate((x_t, x_train), axis = 0)
    y_train_1 = np.concatenate((y_t, np.zeros(y_train)), axis = 0)
    y_train_2 = np.concatenate((np.zeros(y_t), y_train), axis = 0)
    x_val_new = np.concatenate((x_val, x_val), axis = 0)
    y_val_1 = np.concatenate((y_val, np.zeros(y_val)), axis = 0)
    y_val_2 = np.concatenate((np.zeros(y_val), y_val), axis = 0)
    return x_train_new, y_train_1, y_train_2, x_val_new, y_val_1, y_val_2


def xsc_pad(x_train, y_train, x_val, y_val, cfg):
    """ Return the training and validation data of the dysarthric corpus + cross-corpus, if xSC AAI model is chosen.
    
    Parameters
    ----------
    x_train: list 
        Acoustic features of the dysarthric corpus, for training. 
    y_train: list 
        Articulatory features, for training. 
    x_val: list 
        Acoustic features, for model validation
    y_val: list 
        Articulatory features, for validation. 
    cfg: main.Configuration
        Configuration file

    Returns
    -------
    x_train_new: list
        Cross-corpus + Dysarthric corpus training data
    y_train_new: list 
        Cross-corpus + Dysarthic corpus articulatory trajectories.
    x_val: list
        Validation data from the dysarthric corpus only.
    y_val: list
        8-dim trajectories of the dysarthric corpus by masking the other 8-dims. 
    e_t: list 
        x-vectors from the cross-corpus.
    """
    x_t, y_t, e_t, x_v, y_v, e_v = Get_SPIRE_data(cfg) # obtain data from the cross-corpus
    x_train_new = np.concatenate((x_t, x_train), axis = 0)
    y_train_new = np.concatenate((y_t, y_train), axis = 0)
    return x_train_new, y_train_new, x_val, y_val, e_t


def xmc_pad(x_train, y_train, x_val, y_val, cfg):
    """ Return the training and validation data of the dysarthric corpus + cross-corpus, if xMC AAI model is chosen.
    
    Parameters
    ----------
    x_train: list 
        Acoustic features of the dysarthric corpus, for training. 
    y_train: list 
        Articulatory features, for training. 
    x_val: list 
        Acoustic features, for model validation
    y_val: list 
        Articulatory features, for validation. 
    cfg: main.Configuration
        Configuration file

    Returns
    -------
    x_train_new: list
        Cross-corpus + Dysarthric corpus training data
    y_train_1: list 
        Cross-corpus' articulatory trajectories by masking the dysarthric corpus's trajectories.
    y_train_2: list
        Dysarthric corpus' articulatory trajectories by masking the cross-corpus' trajectories.
    x_val_new: list
        Validation data from the dysarthric corpus only.
    y_val_1: list
        8-dim trajectories of the dysarthric corpus by masking the other 8-dims. 
    y_val_2: list
        8-dim trajectories of the dysarthric corpus by masking the other 8-dims. 
    e_t: list 
        x-vectors from the cross-corpus.
    """
    x_t, y_t, e_t, x_v, y_v, e_v = Get_SPIRE_data(cfg) # obtain data from the cross-corpus
    x_train_new = np.concatenate((x_t, x_train), axis = 0)
    y_train_1 = np.concatenate((y_t, np.zeros(y_train)), axis = 0)
    y_train_2 = np.concatenate((np.zeros(y_t), y_train), axis = 0)
    x_val_new = np.concatenate((x_val, x_val), axis = 0)
    y_val_1 = np.concatenate((y_val, np.zeros(y_val)), axis = 0)
    y_val_2 = np.concatenate((np.zeros(y_val), y_val), axis = 0)
    return x_train_new, y_train_1, y_train_2, x_val_new, y_val_1, y_val_2, e_t
