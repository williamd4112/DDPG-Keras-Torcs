from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from CriticNetwork import CriticNetwork
import numpy as np

sess = tf.Session()
critic = CriticNetwork(sess, (4, 64, 64), 3, 32, 0.5, 0.5, True)

# Test predict
test_predict_s = np.random.random([4, 64, 64])
test_predict_a = np.random.random([3])
print(critic.model.input_shape)
res = critic.model.predict([test_predict_s.reshape((-1, 4, 64, 64)), test_predict_a.reshape((-1, 3))])
print('Predict pass', res)

# Test gradient
test_gradient_s = np.random.random([4, 64, 64]).reshape((-1, 4, 64, 64))
test_gradient_a = np.random.random([3]).reshape((-1, 3))
grad = critic.gradients(test_gradient_s, test_gradient_a)
print('Gradients pass', grad)

# Test target train
critic.target_train()
print('Target train pass')
