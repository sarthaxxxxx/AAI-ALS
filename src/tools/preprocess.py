import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def Get_Wav_EMA_PerFile(ema, mfcc, label, cfg, tt_max = 200):
    """Return mean and variance normalised ema, mfcc, and x-vectors if required. 

    Parameters
    ----------
    ema: list   
        8-dim articulatory features
    mfcc: list 
        39-dim acoustic features
    label: str 
        path of a label transcription file
    cfg: main.Configuration
        configuration fils
    tt_max: int
        Since features are sampled at 100 Hz, speech chunks of 2 s (200) were chosen.

    Returns
    -------
    If x-vectors are required, return preprocessed ema and mfcc features along w the x-vectors for each speech chunk.
    Else, return preprocessed ema and mfcc features.
    """

    MeanOfData_EMA = np.mean(ema, axis=0) 
    Ema_temp2 = ema - MeanOfData_EMA
    C = 0.5 * np.sqrt(np.mean(np.square(Ema_temp2), axis=0))  
    Ema = np.divide(Ema_temp2, C) # Mean & variance normalised

    MeanOfData_MFCC = np.mean(mfcc, axis=0) 
    mfcc -= MeanOfData_MFCC
    D = 0.5 * np.sqrt(np.mean(np.square(mfcc), axis=0))
    Mfcc = np.divide(mfcc, D)

    ema_final = [] ; mfcc_final = [] 
    
    file_l = os.path.join(label[0:label.find('.') + 4])
    file = np.asarray(np.loadtxt(file_l, usecols = (0,1))) # load label transcription file
    if (len(file.shape)) == 1:
        file = file.reshape(1,-1)

    if cfg.x_vectors:
        XV = []
        splits = label.split('/')
        xvec_path = str(cfg.data_dir + '/'.join(splits[5:8]) + '/Xvector_Kaldi/')
        x_vec = np.loadtxt(os.path.join(xvec_path, splits[-1])) # load 512 dim x-vectors, if required. 

    for i in range(len(file)):
        start_index = int((file[i][0]) * 100) # starting point of a speech segment
        end_index = int((file[i][1]) * 100) # ending point of the segment
        
        if end_index == Ema.shape[0]:
            end_index -= 1
            x = Ema[start_index: end_index,:]
            y = Mfcc[start_index: end_index + 1,:]
        else:
            x = Ema[start_index: end_index,:]
            y = Mfcc[start_index: end_index,:]
        
        if end_index - start_index > tt_max: # if segment duration > 2 s
            s = 0; en = tt_max
            while en < len(x):
                ema_final.append(x[s: en,:])
                mfcc_final.append(y[s: en,:])
                if cfg.x_vectors: XV.append(x_vec)
                if len(x) - en < tt_max: # remaining chunk duration (to achieve 2 s)
                    pad_zeroes_length = tt_max - (len(x)-en)
                    e = np.vstack((x[en::], np.zeros((pad_zeroes_length, cfg.ema_dim), dtype = float)))
                    m = np.vstack((y[en::], np.zeros((pad_zeroes_length, cfg.mfcc_dim), dtype = float)))
                    ema_final.append(e)
                    mfcc_final.append(m)
                    if cfg.x_vectors: XV.append(x_vec)

                s += tt_max
                en += tt_max
        else: # if segment duration < 2 s
            ema_final.append(np.vstack((x, np.zeros((tt_max - len(x), cfg.ema_dim), dtype = float))))
            mfcc_final.append(np.vstack((y, np.zeros((tt_max - len(y), cfg.mfcc_dim), dtype = float))))
            if cfg.x_vectors: XV.append(x_vec)

    if cfg.x_vectors:return ema_final, mfcc_final, XV
    return ema_final, mfcc_final



def pad(x_train, y_train, x_val, y_val, tt_max = 200):
    """ Return sequences padded to the same length.

    Parameters
    ----------
    x_train: list 
        Acoustic features (trainable)
    y_train: list
        Articulatory features (trainable)
    x_val: list
        Acoustic features (for model validation)
    y_val: list
        Articulatory features (for model validation)

    Returns
    -------
    Returns train and validation data of the dysarthric corpus, by padding sequences to the same length.
    """
    x_train = pad_sequences(x_train, padding = 'post', 
                            maxlen = tt_max, dtype = 'float')
    y_train = pad_sequences(y_train, padding = 'post', 
                            maxlen = tt_max, dtype = 'float')
    x_val = pad_sequences(x_val, padding = 'post', 
                          maxlen = tt_max, dtype = 'float')
    y_val = pad_sequences(y_val, padding = 'post', 
                          maxlen = tt_max, dtype = 'float')
    return x_train, y_train, x_val, y_val
