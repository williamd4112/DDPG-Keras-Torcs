import numpy as np
import keras.backend as K


def init_output_layer(shape, name=None):
    value = np.random.uniform(-3e-4, 3e-4, shape)
    return K.variable(value, name=name)

