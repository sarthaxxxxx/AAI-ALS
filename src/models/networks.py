import keras
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, Dense, TimeDistributed 


def ri(cfg):
    """ Return the RI (randomly initialised) AAI model, trained on the dysarthric corpus only.
    3 BLSTM layers, and one single linear regression output layer to obtain articulatory trajectories. 

    Parameters
    ----------
    cfg: main.Configuration
        user configuration file

    Returns
    -------
    Model
    """

    mdninput_Lstm = keras.Input(shape = (None, cfg.mfcc_dim))
    lstm_1 = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(mdninput_Lstm)
    lstm_2a = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                            return_sequences=True))(lstm_1)
    lstm_2 = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(lstm_2a)
    output = TimeDistributed(Dense(cfg.ema_dim, activation='linear'))(lstm_2)
    model = keras.models.Model(mdninput_Lstm, output)
    return model


def mc(cfg):
    """ Return the MC (multi-corpus) AAI model, trained on the dysarthric and cross corpora.
    3 BLSTM layers, and two single linear regression output layers to obtain articulatory trajectories corresponding to each corpus. 

    Parameters
    ----------
    cfg: main.Configuration
        user configuration file

    Returns
    -------
    Model
    """
    mdninput_Lstm = keras.Input(shape = (None, cfg.mfcc_dim))
    lstm_1 = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(mdninput_Lstm)
    lstm_2a = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                            return_sequences=True))(lstm_1)
    lstm_2 = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(lstm_2a)
    output_1 = TimeDistributed(Dense(cfg.ema_dim, 
                                     activation='linear'))(lstm_2)
    output_2 = TimeDistributed(Dense(cfg.ema_dim, 
                                     activation='linear'))(lstm_2)
    model = keras.models.Model(mdninput_Lstm, [output_1, output_2])
    return model


def xsc(cfg):
    """ Return the xSC (speaker-conditioned (x-vectors)) AAI model, trained on the dysarthric and cross corpora.
    3 BLSTM layers, and one single linear regression output layers to obtain articulatory trajectories corresponding to each corpus. 
    Input acoustic features are conditioned w/ the 512-dim x-vectors. 

    Parameters
    ----------
    cfg: main.Configuration
        user configuration file

    Returns
    -------
    Model
    """
    mdninput_Lstm = keras.Input(shape = (None, cfg.mfcc_dim))
    spk_emb = keras.Input(shape = (None, cfg.x_vectors_dim))
    MFCCinput_Lstm = TimeDistributed(Dense(200))(mdninput_Lstm)
    SpkEmb_Lstm = TimeDistributed(Dense(32))(spk_emb)
    mfcc_spk_emb = keras.layers.concatenate([MFCCinput_Lstm, 
                                             SpkEmb_Lstm], axis = -1)
    lstm_1 = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(mfcc_spk_emb)
    lstm_2a = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(lstm_1)
    lstm_2 = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(lstm_2a)
    output = TimeDistributed(Dense(cfg.ema_dim, activation='linear'))(lstm_2)
    model = keras.models.Model([mdninput_Lstm, spk_emb], output)
    return model


def xmc(cfg):
    """ Return the xMC (multi-corpus + speaker-conditioned (x-vectors)) AAI model, trained on the dysarthric and cross corpora.
    3 BLSTM layers, and two linear regression output layers to obtain articulatory trajectories corresponding to each corpus. 
    Input acoustic features are conditioned w/ the 512-dim x-vectors. 
    
    Parameters
    ----------
    cfg: main.Configuration
        user configuration file

    Returns
    -------
    Model
    """
    mdninput_Lstm = keras.Input(shape = (None, cfg.mfcc_dim))
    spk_emb = keras.Input(shape = (None, cfg.x_vectors_dim))
    MFCCinput_Lstm = TimeDistributed(Dense(200))(mdninput_Lstm)
    SpkEmb_Lstm = TimeDistributed(Dense(32))(spk_emb)
    mfcc_spk_emb = keras.layers.concatenate([MFCCinput_Lstm, 
                                             SpkEmb_Lstm], axis = -1)
    lstm_1 = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(mfcc_spk_emb)
    lstm_2a = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(lstm_1)
    lstm_2 = Bidirectional(CuDNNLSTM(cfg.hyperparameters['BLSTM_units'], 
                           return_sequences=True))(lstm_2a)
    output_1 = TimeDistributed(Dense(cfg.ema_dim, activation='linear'))(lstm_2)
    output_2 = TimeDistributed(Dense(cfg.ema_dim, activation='linear'))(lstm_2)
    model = keras.models.Model([mdninput_Lstm, spk_emb], 
                               [output_1, output_2])
    return model



