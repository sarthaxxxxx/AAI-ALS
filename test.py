import os
import numpy as np
import argparse
from scipy.io import loadmat, savemat

from utils.config_tool import Configuration
from src.tools import *

parser = argparse.ArgumentParser(description = 'CLI args for testing')
parser.add_argument('--config', 
                    type = str,
                    default = '/home/sarthak/als_aai/cfg.yaml',
                    help = 'User configuration file')
cfg = parser.parse_args()

def test_fetch(cfg, fold, sub_cond):
    """Return test data for a certain fold, based on the subject conditions.

    Parameters
    ----------
    cfg: main.Configuration
        Configuration file
    fold: int
        current k-fold
    sub_cond: str
        seen/unseen subject conditions

    Returns
    -------
    A list containing paths to diff test labels, depending on the fold.
    """
    if sub_cond == 'seen':
        print('---Loading test data for seen subject conditions---')
        test_files = list(loadmat(os.path.join(cfg.data_loc,
                                               'seen_' + str(fold + 1) + '_test.mat'))['test'])
        return test_files
    print('---Loading test data for unseen subject conditions---')
    return list(loadmat(os.path.join(cfg.data_loc,
                                               'unseen_' + str(fold + 1) + '_test.mat'))['test'])


def hcp_split(data):
    """ Return label paths for both controls and patients. 

    Parameters
    ----------
    data: list
        List of label paths.

    Returns
    -------
    Returns seperate lists for controls and patients.
    """
    control = []; patient = [];
    [control.append(file[0: file.find('.') + 4]) if file.find('Control') > 0 
    else patient.append(file[0: file.find('.') + 4])
    for file in data]
    return control, patient


def main(config):
    """ Main module to read test data, pre-process and begin the testing phase based on user configuration.

    Parameters
    ----------
    config: main.Configuaration
        Configuration file 
    """
    
    assert isinstance(config, object)
    cfg = Configuration(config)

    cc_c_t1 = np.array([], dtype = np.float32).reshape(0, cfg.ema_dim)
    cc_c_t2 = np.array([], dtype = np.float32).reshape(0, cfg.ema_dim)
    cc_c_t3 = np.array([], dtype = np.float32).reshape(0, cfg.ema_dim)
    cc_p_t1 = np.array([], dtype = np.float32).reshape(0, cfg.ema_dim)
    cc_p_t2 = np.array([], dtype = np.float32).reshape(0, cfg.ema_dim)
    cc_p_t3 = np.array([], dtype = np.float32).reshape(0, cfg.ema_dim)

    for fold in range(cfg.folds):
        print(f"Fold: {fold + 1}")
        test_data = test_fetch(cfg, fold, cfg.sub_conds) # fetch all test data
        control, patient = hcp_split(test_data) # seperate controls and patients

        for (c, p) in zip(control, patient):
            (ema_c, mfcc_c) , (ema_p, mfcc_p) = fetch(c, cfg.data_dir), fetch(p, cfg.data_dir) # get pre-processed ema and mfcc
            cc_per_task_c, cc_per_task_p = preds(ema_c, mfcc_c, c, cfg, fold), preds(ema_p, mfcc_p, p, cfg, fold) # obtain cc for each label
            label_c, label_p = c.split('/')[-1], p.split('/')[-1]
            # to avoid any erroneous cases, report cc per speech task
            if label_c.split('_')[0] == 'T1' or label_c.split('_')[0] == 'T2': cc_c_t1 = np.vstack([cc_c_t1, cc_per_task_c])
            if label_p.split('_')[0] == 'T1' or label_p.split('_')[0] == 'T2': cc_p_t1 = np.vstack([cc_p_t1, cc_per_task_p])
            if label_c.split('_')[0] == 'T4': cc_c_t2 = np.vstack([cc_c_t2, cc_per_task_c])
            if label_p.split('_')[0] == 'T4': cc_p_t2 = np.vstack([cc_p_t2, cc_per_task_p])
            if label_c.split('_')[0] == 'T5': cc_c_t3 = np.vstack([cc_c_t3, cc_per_task_c])
            if label_p.split('_')[0] == 'T5': cc_p_t3 = np.vstack([cc_p_t3, cc_per_task_p])
            
    total_c, total_p = np.array([], dtype = np.float32).reshape(0, cfg.ema_dim), np.array([], dtype = np.float32).reshape(0, cfg.ema_dim)
    total_c, total_p = np.vstack([cc_c_t1, cc_c_t2, cc_c_t3]), np.vstack([cc_p_t1, cc_p_t2, cc_p_t3])

    print('---Results for healthy controls---') # average across all articulators and folds
    print(f"T1 w/ mean {np.mean(np.mean(cc_c_t1, axis = 0))} and std {np.std(np.mean(cc_c_t1, axis = 0))}")
    print(f"T2 w/ mean {np.mean(np.mean(cc_c_t2, axis = 0))} and std {np.std(np.mean(cc_c_t1, axis = 0))}")
    print(f"T3 w/ mean {np.mean(np.mean(cc_c_t3, axis = 0))} and std {np.std(np.mean(cc_c_t1, axis = 0))}")
    print(f"Average CC w/ mean {np.mean(np.mean(total_c, axis = 0))} and std of {np.std(np.mean(total_c, axis = 0))}")

    #Likewise for patients


if __name__ == '__main__':
    main(cfg.config)
