import os 
import numpy as np
from scipy.io import loadmat


def fetch(label_path, data_dir):
    """ Return the articulatory(ema) and acoustic(mfcc) data for a certain speech wave, with its label path given as an input. 

    Parameters
    ----------
    label_path: str
        path of a label transcription file (has manually labelled speech segments)
    data_dir: str
        path to the dysarthric corpus

    Returns
    -------
    ema: list
        8-dim articulatory features 
    mfcc: list
        39-dim MFCC
    """

    splits = label_path.split('/')
    ema_dir = str(data_dir + '/'.join(splits[5:8]) + "/EmaData/")
    mfcc_dir = str(data_dir + '/'.join(splits[5:8]) + "/MFCC_Kaldi/")
    
    ema_name = str((splits[-1].split('.'))[0][:-4] + 
               'EMA_' + (splits[-1].split('.'))[0][-4:] + '.mat') # extract ema filename for a certain label file name
    ema = loadmat(os.path.join(ema_dir, ema_name))['EmaData'][6:] # remove articulatory movements in the z directions of 6 articulators collected
    ema = np.transpose(np.delete(ema, [1,4,7,10,12,13,14,15,16,17], 0)) # consider x and y directions of only 4 articulators (resultant order: ULx, ULy, LLx, LLy, JAWx, JAWy, TTx, TTy)
    mfcc = (np.loadtxt(os.path.join(mfcc_dir, splits[-1]))) # load the mfcc features
    return ema, mfcc
