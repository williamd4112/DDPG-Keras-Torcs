from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from ActorNetwork_fusion import ActorNetwork
from CriticNetwork import CriticNetwork
import numpy as np

sess = tf.Session()
actor = ActorNetwork(sess, (64, 64, 4), (29,), 3, 32, 0.5, 0.5, True)
#critic = CriticNetwork(sess, (64, 64, 4), 3, 32, 0.5, 0.5, True)
print('Create pass')

# Test predict
test_predict_vision = np.random.random([64, 64, 4])
test_predict_sensor = np.random.random([29])
res = actor.model.predict([test_predict_vision.reshape((-1, 64, 64, 4)), test_predict_sensor.reshape((-1, 29))])
print('Predict pass', res)

# Test train
test_train_vision = np.random.random([64, 64, 4])
test_train_sensor = np.random.random([29])
act_for_grad = actor.model.predict([test_train_vision.reshape((-1, 64, 64, 4)), test_train_sensor.reshape((-1, 29))]).reshape((-1, 3))
actor.train(test_train_vision.reshape((-1, 64, 64, 4)), test_train_sensor.reshape((-1, 29)), act_for_grad)
print('Train pass', act_for_grad)

# Test target train
actor.target_train()
print('Target train pass')


