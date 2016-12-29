import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Convolution2D, Permute
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
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, vision):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size, vision)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size, vision)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim, vision):
        print("Now we build the model")
        if vision:
            S = Input(shape=state_size)
            A = Input(shape=[action_dim],name='action2')

            input_s = Permute((2, 3, 1))(S)
            conv_1 = Convolution2D(32, 8, 8, activation='relu', init='he_uniform')(input_s)
            conv_2 = Convolution2D(32, 8, 8, activation='relu', init='he_uniform')(conv_1)
            conv_3 = Convolution2D(32, 8, 8, activation='relu', init='he_uniform')(conv_2)
            conv_flat = Flatten()(conv_3)
            hidden_s_1 = Dense(HIDDEN1_UNITS, activation='relu', init='he_uniform')(conv_flat)
            hidden_s_2 = Dense(HIDDEN2_UNITS, activation='relu', init='he_uniform')(hidden_s_1)

            hidden_a_1 = Dense(HIDDEN2_UNITS, activation='linear', init='he_uniform')(A)

            hidden_sa_0 = merge([hidden_s_2, hidden_a_1], mode='sum')

            hidden_sa_1 = Dense(HIDDEN2_UNITS, activation='relu', init='he_uniform')(hidden_sa_0)

            V = Dense(action_dim, activation='tanh', init=init_output_layer)(hidden_sa_1)

            model = Model(input=[S,A],output=V)
            adam = Adam(lr=self.LEARNING_RATE)
            model.compile(loss='mse', optimizer=adam)
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
