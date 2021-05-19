import os 
import numpy as np
from keras.models import load_model
from src.metric import *


def preds(ema, mfcc, label, cfg, fold):
    """ Return Pearson correlation coefficent for a certain speech task, with the label path given as an input.

    Parameters
    ----------
    ema: list 
        EMA data for a speech file (8-dims)
    mfcc: list 
        MFCC data (39-dims)
    label: str
        Path to label file. Use this to exract ema and mfcc. 
    cfg: main.Configuration
        Configuration file
    fold: int
        Current k-fold

    Returns
    -------
    CC for a certain speech task.
    """
    MeanOfData_EMA = np.mean(ema, axis=0) 
    Ema_temp2 = ema - MeanOfData_EMA
    C = 0.5 * np.sqrt(np.mean(np.square(Ema_temp2), axis=0))  
    Ema = np.divide(Ema_temp2, C) # Mean & variance normalised

    file_l = os.path.join(label[0:label.find('.') + 4])
    file = np.asarray(np.loadtxt(file_l, usecols = (0,1)))
    if (len(file.shape)) == 1:
        file = file.reshape(1,-1)

    if cfg.x_vectors:
        splits = label.split('/')
        xvec_path = str(cfg.data_dir + '/'.join(splits[5:8]) + '/Xvector_Kaldi/')
        x_vec = np.loadtxt(os.path.join(xvec_path, splits[-1]))

    cc_per_task = np.array([], dtype = np.float32).reshape(0, cfg.ema_dim)

    for i in range(len(file)):
        start_index = int((file[i][0]) * 100)
        end_index = int((file[i][1]) * 100)
        
        if end_index == Ema.shape[0]:
            end_index -= 1
            ema_gt = Ema[start_index:end_index,:]
            mfcc_gt = mfcc[start_index:end_index + 1,:]
        else:
            ema_gt = Ema[start_index:end_index,:]
            mfcc_gt = Ema[start_index:end_index + 1,:]
        M_t = mfcc_gt[np.newaxis,:,:]
        if cfg.x_vectors: # if xvecs are used, perform prediction (arti trajectories) accordingly
            xvec_emb = np.ones((1, M_t.shape[1], 1), order = 'C') * x_vec
            PredEMA = model_pred(cfg, M_t, fold, xvec_emb)
        else: PredEMA = model_pred(cfg, M_t, fold, xvec_emb = None)
        if ema_gt.shape[0] != PredEMA.shape[0]:
            ema_gt = ema_gt[0:PredEMA.shape[0],:]
        cc = eval_metric(ema_gt, PredEMA[:,:,cfg.ema_dim]) # obtain CC
        cc = np.around(np.reshape(cc, (1, -1)), decimals = 4) # round off
        cc_per_task = np.vstack([cc_per_task, cc]) # store cc for each task
        return cc_per_task



def model_pred(cfg, mfcc, fold, x_vectors):
    """ Load the required best model and return the prediction (articulatory trajectories).

    Parameters
    ----------
    cfg: main.Configuration
        Configuration file
    mfcc: list 
        Input features for the model, for prediction
    fold: int
        Current k-fold
    x-vectors: list
        x-vectors, if required

    Returns
    -------
    Return predictions from the model
    """

    file_name = cfg.model_dir + str(cfg.hyperparameters['model']) + '_' + str(cfg.sub_conds) + '_' + str(fold + 1) + '_ALS_AAI_.h5' # model name based on cfg
    if cfg.hyperparameters['model'].lower() == 'ri':
        model = load_model(file_name) 
        return np.squeeze(model.predict(mfcc))
    elif cfg.hyperparameters['model'].lower() == 'mc':
        model = load_model(file_name, compile = False)
        PredEMA_cross, PredEMA_dys = model.predict(mfcc) # consider predictions(arti trajectories) from the dysarthric corpus
        return np.squeeze(PredEMA_dys)
    elif cfg.hyperparameters['model'].lower() == 'xsc':
        model = load_model(file_name)
        return np.squeeze(model.predict[mfcc, x_vectors]) # condition input acoustics w/ x-vectors
    else:
        model = load_model(file_name, compile = False)
        PredEMA_cross, PredEMA_dys = model.predict([mfcc, x_vectors])
        return np.squeeze(PredEMA_dys) # consider preds(arti trajectories) from the dysarthric corpus
