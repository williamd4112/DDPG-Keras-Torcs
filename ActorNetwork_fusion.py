import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, TimeDistributedDense, Flatten, Input, merge, Lambda, Permute, Convolution2D, Activation
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

from Common import *

#HIDDEN1_UNITS = 300
#HIDDEN2_UNITS = 600

HIDDEN1_UNITS = 200
HIDDEN2_UNITS = 200
HIDDEN3_UNITS = 64

class ActorNetwork(object):
    def __init__(self, sess, state_vision_size, state_sensor_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, vision, summary_writer=None):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state_vision, self.state_sensor = self.create_actor_network(state_vision_size, state_sensor_size, action_size, vision)
        self.target_model, self.target_weights, self.target_state_vision, self.target_state_sensor = self.create_actor_network(state_vision_size, state_sensor_size, action_size, vision)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, state_vision, state_sensor, action_grads):
        _ = self.sess.run([self.optimize], feed_dict={
            self.state_vision: state_vision,
            self.state_sensor: state_sensor,
            self.action_gradient: action_grads
        })
        #self.summary_writer.add_summary(summary)

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_vision_size, state_sensor_size, action_dim, vision):
        print("Now we build the model")
        if vision:
            input_vision = Input(shape=state_vision_size)
            conv_1 = Convolution2D(32, 3, 3, subsample=(2, 2), activation='relu', init='he_uniform')(input_vision)
            #pool_1 = MaxPooling2D()(conv_1)
            conv_2 = Convolution2D(32, 3, 3, subsample=(2, 2), activation='relu', init='he_uniform')(conv_1)
            #pool_2 = MaxPooling2D()(conv_2)
            conv_3 = Convolution2D(32, 3, 3, subsample=(2, 2), activation='relu', init='he_uniform')(conv_2)
            #pool_3 = MaxPooling2D()(conv_3)

            flat = Flatten()(conv_3)

            hidden_1 = Dense(HIDDEN1_UNITS, activation='relu', init='he_uniform')(flat)
            hidden_2 = Dense(HIDDEN2_UNITS, activation='relu', init='he_uniform')(hidden_1)

            input_sensor = Input(shape=state_sensor_size)
            hidden_sensor_0 = Dense(HIDDEN1_UNITS, activation='relu')(input_sensor)
            hidden_sensor_1 = Dense(HIDDEN2_UNITS, activation='relu')(hidden_sensor_0)

            hidden_fusion = merge([hidden_2, hidden_sensor_1], mode='concat')
            hidden_fusion_reshape = Reshape((1, HIDDEN2_UNITS * 2))(hidden_fusion)
            hidden_fusion_lstm = LSTM(HIDDEN2_UNITS, return_sequences=False)(hidden_fusion_reshape)
            hidden_fusion_fc = Dense(HIDDEN3_UNITS, activation='relu')(hidden_fusion)

            Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-6, name=name))(hidden_fusion_lstm)
            Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-6, name=name))(hidden_fusion_lstm)
            Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(hidden_fusion_lstm)

            #V = Dense(action_dim, activation='tanh', init=init_output_layer)(hidden_2)
            V = merge([Steering,Acceleration,Brake],mode='concat')

            model = Model(input=[input_vision, input_sensor],output=V)
            return model, model.trainable_weights, input_vision, input_sensor
        else:
            S = Input(shape=[state_size])
            h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
            h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
            Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
            Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
            Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
            V = merge([Steering,Acceleration,Brake],mode='concat')
            model = Model(input=S,output=V)
            return model, model.trainable_weights, S
