from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from CriticNetwork_fusion import CriticNetwork
import numpy as np

sess = tf.Session()
critic = CriticNetwork(sess, (64, 64, 4), (29,), 3, 32, 0.5, 0.5, True)

# Test predict
test_predict_s_vision = np.random.random([64, 64, 4]).reshape((-1, 64, 64, 4))
test_predict_s_sensor = np.random.random([29]).reshape((-1, 29))
test_predict_a = np.random.random([3]).reshape((-1, 3))
res = critic.model.predict([test_predict_s_vision, test_predict_s_sensor, test_predict_a])
print('Predict pass', res)

# Test gradient
test_gradient_s_vision = np.random.random([64, 64, 4]).reshape((-1, 64, 64, 4))
test_gradient_s_sensor = np.random.random([29]).reshape((-1, 29))
test_gradient_a = np.random.random([3]).reshape((-1, 3))
grad = critic.gradients(test_gradient_s_vision, test_gradient_s_sensor, test_gradient_a)
print('Gradients pass', grad)

# Test target train
critic.target_train()
print('Target train pass')
