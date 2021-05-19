import numpy as np
from scipy.stats import pearsonr


def eval_metric(X, Y, cfg):
    """ To measure correlation between the ground-truth and the predicted articulatory trajectories.
    
    Parameters
    ----------
    X: list
        Ground-truth articulatory trajectories
    Y: list 
        Predicted articulatory trajectories
    cfg: main.Configuration
        Config file
        
    Returns
    -------
    Correlation for each articulator (8 dims)
    """
    
    CC = [pearsonr(X[:,idx], Y[:,idx]) for idx in range(cfg.ema_dim)]
    # rmse = np.sqrt(np.mean(np.square(X - Y), axis = 0))
    return np.array(CC)[:,0]
