import numpy as np
import scipy.io
import os
import sys
import numpy
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from scipy.stats import pearsonr


DataDir = '/home2/data/ARAVIND/End2End/SPIRE_EMA/DataBase/'
Trainsubs = sorted(os.listdir(DataDir))

def Get_Wav_EMA_PerFile(EMA_file, Wav_file, F, EmaDir, MFCCpath, BeginEnd, XvectorPath, cfg):
    """Return mean and variance normalised ema, mfcc, and x-vectors if required (of the cross-corpus). 

    Parameters
    ----------
    EMA_file: str 
        path to ema file
    Wav_file: str 
        path to mfcc file
    F: int 
        sentence number from MOCHA-TIMIT
    EMADir: str
        Directory to EMA data
    MFCCpath: str 
        Directory to MFCC data
    BeginEnd: array
        Start and end of a speech segment
    XvectorPath: str
        Path to xvector directory
    cfg: main.Configuration
        Configuration file

    Returns
    -------
    If x-vectors are required, return preprocessed ema and mfcc features along w the x-vectors for each speech chunk.
    Else, return preprocessed ema and mfcc features.
    """

    EmaMat = scipy.io.loadmat(EmaDir + EMA_file)
    EMA_temp = EmaMat['EmaData']
    EMA_temp = np.transpose(EMA_temp)# time X 18
    Ema_temp2 = np.delete(EMA_temp, [4,5,6,7,10,11,14,15,16,17], 1) # time X 12
    MeanOfData = np.mean(Ema_temp2, axis=0)
    Ema_temp2 -= MeanOfData
    C = 0.5*np.sqrt(np.mean(np.square(Ema_temp2), axis=0))
    Ema = np.divide(Ema_temp2, C) # Mean & variance normailized
    [aE,bE] = Ema.shape

    #print F.type
    EBegin = np.int(BeginEnd[0,F]*100) # start of a speech segment
    EEnd = np.int(BeginEnd[1,F]*100) # end of the segment
    feats = np.loadtxt(MFCCpath + Wav_file[:-4] + '.txt')#np.asarray(htkfile.data)
    MFCC_G = feats
    TimeStepsTrack = EEnd - EBegin
    ## X-vector Embeddings
    SPK_emd = np.loadtxt(XvectorPath + Wav_file[:-4] + '.txt') # load x-vectors, given a wav file name

    if cfg.x_vectors:
        return Ema[EBegin:EEnd,:], MFCC_G[EBegin:EEnd,:cfg.mfcc_dim], SPK_emd # with out 
    return Ema[EBegin:EEnd,:], MFCC_G[EBegin:EEnd,:cfg.mfcc_dim]


RootDir = '/home2/data/ARAVIND/End2End/SPIRE_EMA/'

def Get_SPIRE_data(cfg, TT_max=200):
    """ Return training and validation data of the cross-corpus.

    Parameters
    ----------
    cfg: main.Configuration
        Configuration file.
    TT_max: int 
        Duration of speech chunk (2 sec, since features are sampled at 100 Hz)

    Returns
    -------
    If x-vectors are required, return padded training and validation data w/ the x-vectors.
    Else, return padded training and validation data.
    """

    X_valseq = []; Youtval = [];
    X_trainseq = []; Youttrain = [];
    X_testseq = []; Youttest = [];
    E_valseq = []; E_trainseq = [];
    for ss in np.arange(0, len(Trainsubs)):
        Sub = Trainsubs[ss] 
        print(Sub)
        WavDir = RootDir + 'DataBase/' + Sub + '/Neutral/WavClean/';
        EmaDir = RootDir + 'DataBase/' + Sub + '/Neutral/EmaClean/';
        BeginEndDir = RootDir + '/StartStopMat/' + Sub + '/';
        MFCCpath = RootDir + 'MFCC_Kaldi/' + Sub + '/' #Neutral/MfccHTK/'
        XvectorPath = RootDir + 'Xvector_Kaldi/' + Sub + '/'
        EMAfiles = sorted(os.listdir(EmaDir))
        Wavfiles = sorted(os.listdir(WavDir))
        StartStopFile = os.listdir(BeginEndDir)
        StartStopMAt = scipy.io.loadmat(BeginEndDir+StartStopFile[0])
        BeginEnd = StartStopMAt['BGEN']
        F = 5 # Fold No
        for i in np.arange(0,460): # 460 sentences from MOCHA-TIMIT
            if (((i + F) % 10)==0):# Test # 10% for test
                if cfg.x_vectors: E_t, M_t, se = Get_Wav_EMA_PerFile(EMAfiles[i], Wavfiles[i], i, EmaDir, MFCCpath, BeginEnd, XvectorPath, cfg)
                else: E_t, M_t = Get_Wav_EMA_PerFile(EMAfiles[i], Wavfiles[i], i, EmaDir, MFCCpath, BeginEnd, XvectorPath, cfg)
                E_t = E_t[np.newaxis,:,:]
                M_t = M_t[np.newaxis,:,:]
                Youttest.append(E_t)
                X_testseq.append(M_t)
            elif (((i + F + 1) % 10)==0):# Validation # 10% for val
                if cfg.x_vectors: 
                    E_t, M_t, se = Get_Wav_EMA_PerFile(EMAfiles[i], Wavfiles[i], i, EmaDir, MFCCpath, BeginEnd, XvectorPath, cfg)
                    E_valseq.append(se)
                else: 
                    E_t, M_t = Get_Wav_EMA_PerFile(EMAfiles[i], Wavfiles[i], i, EmaDir, MFCCpath, BeginEnd, XvectorPath, cfg)
                Youtval.append(E_t)
                X_valseq.append(M_t)
            else:#elif (((i+F+2)%10)==0):#else: # Train (80%)
                if cfg.x_vectors: 
                    E_t, M_t, se = Get_Wav_EMA_PerFile(EMAfiles[i], Wavfiles[i], i, EmaDir, MFCCpath, BeginEnd, XvectorPath, cfg)
                    E_trainseq.append(se)
                else: 
                    E_t, M_t = Get_Wav_EMA_PerFile(EMAfiles[i], Wavfiles[i], i, EmaDir, MFCCpath, BeginEnd, XvectorPath, cfg)
                Youttrain.append(E_t)
                X_trainseq.append(M_t)

    X_valseq = pad_sequences(X_valseq, padding = 'post', maxlen = TT_max, dtype = 'float')
    Youtval = pad_sequences(Youtval, padding = 'post', maxlen = TT_max, dtype = 'float')
    X_trainseq = pad_sequences(X_trainseq, padding = 'post', maxlen = TT_max, dtype = 'float')
    Youttrain = pad_sequences(Youttrain, padding = 'post', maxlen = TT_max, dtype = 'float')

    if cfg.x_vectors:
        return X_trainseq, Youttrain, E_trainseq, X_valseq, Youtval, E_valseq
    return X_trainseq, Youttrain, X_valseq, Youtval
