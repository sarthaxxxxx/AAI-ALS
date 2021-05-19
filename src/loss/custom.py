import keras
import keras.backend as K

def custom_loss(true, pred):
    """ A custom loss function where the MSE loss from a corpus backprops the error to update the weights while masking the outputs of the other corpus.
    
    Parameters
    ----------
    true: tensor
        Ground-truth articulatory trajectories
    pred: tensor
        Model predicted articulatory trajectories
    
    Returns
    -------
    Loss
    """
    z = K.zeros(K.shape(true), dtype = 'float32')
    b = K.ones(K.shape(true), dtype = 'float32') - K.cast(K.equal(z, true), dtype = 'float32')
    return (K.sum(b, axis = -1)/8) * keras.losses.mse(true, pred)
