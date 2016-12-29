from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import numpy as np

sess = tf.Session()
actor = ActorNetwork(sess, (4, 64, 64), 3, 32, 0.5, 0.5, True)
critic = CriticNetwork(sess, (4, 64, 64), 3, 32, 0.5, 0.5, True)
print('Create pass')

# Test predict
test_predict = np.random.random([4, 64, 64])
res = actor.model.predict(test_predict.reshape((-1, 4, 64, 64)))
print('Predict pass', res)

# Test train
test_train = np.random.random([4, 64, 64])
test_train_s_1 = np.random.random([4, 64, 64]).reshape((-1, 4, 64, 64))
act_for_grad = actor.model.predict(test_train.reshape((-1, 4, 64, 64))).reshape((-1, 3))
actor.train(test_train_s_1, act_for_grad)
print('Train pass', act_for_grad)

# Test target train
actor.target_train()


