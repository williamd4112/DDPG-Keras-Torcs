import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Convolution2D, Permute
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM

from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

from Common import *
#HIDDEN1_UNITS = 300
#HIDDEN2_UNITS = 600

HIDDEN1_UNITS = 200
HIDDEN2_UNITS = 200

class CriticNetwork(object):
    def __init__(self, sess, state_vision_size, state_sensor_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, vision):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state_vision, self.state_sensor = self.create_critic_network(state_vision_size, state_sensor_size, action_size, vision)
        self.target_model, self.target_action, self.target_state_vision, self.target_state_sensor = self.create_critic_network(state_vision_size, state_sensor_size, action_size, vision)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, state_vision, state_sensor, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state_vision: state_vision,
            self.state_sensor: state_sensor,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_vision_size, state_sensor_size, action_dim, vision):
        print("Now we build the model")
        if vision:
            input_vision = Input(shape=state_vision_size)
            input_sensor = Input(shape=state_sensor_size)
            A = Input(shape=[action_dim],name='action2')
            conv_1 = Convolution2D(32, 3, 3, subsample=(1,1), activation='relu', init='he_uniform')(input_vision)
            pool_1 = MaxPooling2D()(conv_1)
            conv_2 = Convolution2D(32, 3, 3, subsample=(1,1), activation='relu', init='he_uniform')(pool_1)
            pool_2 = MaxPooling2D()(conv_2)
            conv_3 = Convolution2D(32, 3, 3, subsample=(1,1), activation='relu', init='he_uniform')(pool_2)
            pool_3 = MaxPooling2D()(conv_3)
            conv_flat = Flatten()(pool_3)
            hidden_s_1 = Dense(HIDDEN1_UNITS, activation='relu', init='he_uniform')(conv_flat)
            hidden_s_2 = Dense(HIDDEN2_UNITS, activation='relu', init='he_uniform')(hidden_s_1)

            hidden_sensor_0 = Dense(HIDDEN1_UNITS, activation='relu')(input_sensor)
            hidden_sensor_1 = Dense(HIDDEN2_UNITS, activation='relu')(hidden_sensor_0)

            hidden_fusion = merge([hidden_s_2, hidden_sensor_1], mode='concat')
            hidden_fusion_reshape = Reshape((1, HIDDEN2_UNITS * 2))(hidden_fusion)
            hidden_fusion_lstm = LSTM(HIDDEN2_UNITS, return_sequences=False)(hidden_fusion_reshape)

            hidden_a_1 = Dense(HIDDEN2_UNITS, activation='linear', init='he_uniform')(A)

            hidden_sa_0 = merge([hidden_fusion_lstm, hidden_a_1], mode='sum')
            hidden_sa_1 = Dense(HIDDEN2_UNITS, activation='relu', W_regularizer=l2(0.01), init='he_uniform')(hidden_sa_0)
            V = Dense(action_dim, activation='tanh', init=init_output_layer)(hidden_sa_1)

            model = Model(input=[input_vision, input_sensor, A],output=V)
            adam = Adam(lr=self.LEARNING_RATE, decay=0.000001)
            model.compile(loss='mse', optimizer=adam)
            return model, A, input_vision, input_sensor
        else:
            S = Input(shape=[state_size])
            A = Input(shape=[action_dim],name='action2')
            w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
            a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
            h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
            h2 = merge([h1,a1],mode='sum')
            h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
            V = Dense(action_dim,activation='linear')(h3)
            model = Model(input=[S,A],output=V)
            adam = Adam(lr=self.LEARNING_RATE)
            model.compile(loss='mse', optimizer=adam)
            return model, A, S
