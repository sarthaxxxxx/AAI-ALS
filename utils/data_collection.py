import os
from scipy.io import loadmat
from src.tools import *
from utils import *


def data_fetch(cfg, fold, sub_cond):
    """ Prepare training and validation data based on the subject conditions and fold. 
    
    Parameters
    ----------
    cfg: main.Configuration
        configuration file
    fold: int
        current k-fold
    sub_cond: str
        seen/unseen subject conditions
    
    Returns
    -------
    If x-vectors are being used, returns the train (x_train, y_train) and validation (x_val, y_val) data w/ their respective x-vectors(e_train, e_val). 
    Else, returns only the train and validation data. 
    """
    
    if sub_cond == 'seen':
        print('---Loading data for seen subject conditions---')
        train_files = list(loadmat(os.path.join(cfg.data_loc,
                                                'seen_' + str(fold + 1) + '_train.mat'))['train'])
        test_files = list(loadmat(os.path.join(cfg.data_loc,
                                                'seen_' + str(fold + 1) + '_test.mat'))['test'])
    else:
        print('---Loading data for unseen subject conditions---')
        train_files = list(loadmat(os.path.join(cfg.data_loc,
                                                'unseen_' + str(fold + 1) + '_train.mat'))['train'])
        test_files = list(loadmat(os.path.join(cfg.data_loc,
                                                'unseen_' + str(fold + 1) + '_test.mat'))['test'])

    train = []; val = [];
    train = train_files[0 : int(0.9 * len(train_files))] # 90% of train data .
    val = train_files[int(0.9 * len(train_files)):]
    val.extend(test_files[0: int(0.1 * len(test_files))]) # val data = 10% (training + test) data (to ensure no overlap)

    x_train = []; x_val = []; y_train = []; y_val = []; e_train = []; e_val = [];

    print('---Loading training and validation data---')

    for t, v in zip(train, val):
        (ema_t, mfcc_t), (ema_v, mfcc_v) = fetch(str(t[0:np.char.find(t,'.')+4]), cfg.data_dir), # return respective ema(8) and mfcc(39) for a given input label file.
        fetch(str(v[0:np.char.find(t,'.')+4]), cfg.data_dir)
        
        if cfg.x_vectors:
            (e_t, m_t, xv_t), (e_v, m_v, xv_v) = Get_Wav_EMA_PerFile(ema_t, mfcc_t, str(t[0:np.char.find(t,'.')+4]), cfg), # return pre-processed ema, mfcc, and x-vectors.
            Get_Wav_EMA_PerFile(ema_v, mfcc_v, str(v[0:np.char.find(v,'.')+4]), cfg)  
            x_train += m_t; y_train += e_t; e_train += xv_t;  x_val += m_v; y_val += e_v; e_val += xv_v
            continue
        (e_t, m_t), (e_v, m_v) = Get_Wav_EMA_PerFile(ema_t, mfcc_t, str(t[0:np.char.find(t,'.')+4]), cfg),
        Get_Wav_EMA_PerFile(ema_v, mfcc_v, str(v[0:np.char.find(v,'.')+4]), cfg)
        
        x_train += m_t; y_train += e_t; x_val += m_v; y_val += e_v 

    if cfg.x_vectors:
        return x_train, y_train, e_train, x_val, y_val, e_val

    return x_train, y_train, x_val, y_val
