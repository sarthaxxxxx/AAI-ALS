import os
from utils import *
from src import *
from utils.sequences import *
from keras.callbacks import ModelCheckpoint, EarlyStopping


def trainer(x_train, y_train, x_val, y_val, fold, cfg, e_train, e_val):
    """ Based on user configuration, begin training of AAI models.

    Parameters
    ----------
    x_train: list
        training acoustic features from the dysarthic corpus.
    y_train: list
        training articulatory features.
    x_val: list
        validation acoustic features.
    y_val: list
        validation articulatory features.
    fold: int 
        current k-fold
    cfg: main.Configuration
        user configuration file
    e_train: list 
        training x-vectors 
    e_val: list 
        validation x-vectors
    """

    tt_max = 200
    ckpt_name = str(cfg.hyperparameters['model']) + '_' + str(cfg.sub_conds) + '_' + str(fold + 1) + '_ALS_AAI_' # checkpoint name 
    if not os.path.exists(cfg.model_dir):
        os.mkdir(cfg.model_dir)
    checkpointer1 = ModelCheckpoint(filepath = cfg.model_dir + ckpt_name +'.h5',
                                    verbose = 0, save_best_only = True) # save best model
    checkpointer2 = ModelCheckpoint(filepath = cfg.model_dir + ckpt_name + 'weights.h5',
                                    verbose = 0, save_best_only = True, save_weights_only = True) # save best model weights
    earlystop = EarlyStopping(monitor = cfg.hyperparameters['monitor'], 
                              patience = cfg.hyperparameters['patience']) # early stopping criterion (based on val loss)
    
    print(f"---Loading {cfg.hyperparameters['model']} AAI model---")

    model = eval(cfg.hyperparameters['model'].lower())(cfg) # load model based on configuration

    if e_train and e_val is not None: # if x-vectors are required and not none
        
        if cfg.hyperparameters['model'].lower() == 'xmc': # xMC Model
            x_t, y_t1, y_t2, x_v, y_v1, y_v2, e_t = eval(str(cfg.hyperparameters['model']).lower() + str('_pad'))
            (x_train, y_train, x_val, y_val, cfg) # obtain final data, including cross-corpus
            e_t_new = np.concatenate((np.array(e_t), np.array(e_train)), axis = 0) # combine x-vec from both corpora
            e_v_new = np.concatenate((np.array(e_val), np.array(e_val)), axis = 0) # validation data only on dysarthric corpus auxillary features
            model.compile(optimizer = 'adam', loss = custom_loss) # use the custom loss function for training
            model.fit(x = [x_t, np.repeat(np.array(e_t_new)[:,np.newaxis,:], tt_max, axis=1)], 
                      y = [y_t1, y_t2],
                      validation_data = ([x_v, np.repeat(np.array(e_v_new)[:,np.newaxis,:], tt_max, axis=1)], [y_v1, y_v2]), 
                      epochs = cfg.hyperparameters['epochs'], 
                      batch_size = cfg.hyperparameters['batch_size'], 
                      verbose = 1, 
                      shuffle = True, 
                      callbacks = [checkpointer1, checkpointer2, earlystop])
        else: #xSC model
            x_t, y_t, x_v, y_v, e_t= eval(str(cfg.hyperparameters['model']).lower() + str('_pad'))
            (x_train, y_train, x_val, y_val, cfg)
            e_t_new = np.concatenate((np.array(e_t), np.array(e_train)), axis = 0)
            model.compile(optimizer = 'adam', loss = 'mse')
            model.fit(x = [x_t, np.repeat(np.array(e_t_new)[:,np.newaxis,:], tt_max, axis=1)], 
                      y = y_t, 
                      validation_data = ([x_v, np.repeat(np.array(e_val)[:,np.newaxis,:], tt_max, axis=1)], y_v), 
                      epochs = cfg.hyperparameters['epochs'],
                      batch_size = cfg.hyperparameters['batch_size'], 
                      verbose = 1, 
                      shuffle = True, 
                      callbacks = [checkpointer1, checkpointer2, earlystop])

    elif e_train and e_val is None:   

        if cfg.hyperparameters['model'].lower() == 'mc': # MC model
            x_t, y_t1, y_t2, x_v, y_v1, y_v2 = eval(str(cfg.hyperparameters['model']).lower() + str('_pad'))
            (x_train, y_train, x_val, y_val, cfg)
            model.compile(optimizer = 'adam', loss = custom_loss)
            model.fit(x = x_t, 
                    y = [y_t1, y_t2], 
                    validation_data = (x_v, [y_v1, y_v2]), 
                    epochs = cfg.hyperparameters['epochs'], 
                    batch_size = cfg.hyperparameters['batch_size'], 
                    verbose = 1, 
                    shuffle = True, 
                    callbacks = [checkpointer1, checkpointer2, earlystop])
        else: # RI model
            x_t, y_t, x_v, y_v = eval(str(cfg.hyperparameters['model']).lower() + str('_pad'))
            (x_train, y_train, x_val, y_val)
            model.compile(optimizer = 'adam', loss = 'mse')
            model.fit(x = x_t, 
                    y = y_t, 
                    validation_data = (x_v, y_v), 
                    epochs = cfg.hyperparameters['epochs'], 
                    batch_size = cfg.hyperparameters['batch_size'], 
                    verbose = 1, 
                    shuffle = True, 
                    callbacks = [checkpointer1, checkpointer2, earlystop])
