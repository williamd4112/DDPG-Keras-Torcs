import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, Permute, Convolution2D, Activation

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

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, vision, summary_writer=None):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size, vision)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size, vision)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())
        self.summary_writer = summary_writer

    def train(self, states, action_grads):
        _ = self.sess.run([self.optimize], feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })
        #self.summary_writer.add_summary(summary)

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim, vision):
        print("Now we build the model")
        if vision:
            S = Input(shape=state_size)
            input_s = Permute((1, 2, 3))(S)
            conv_1 = Convolution2D(16, 3, 3, subsample=(1, 1), activation='relu', init='he_uniform')(input_s)
            pool_1 = MaxPooling2D()(conv_1)
            conv_2 = Convolution2D(32, 3, 3, subsample=(1, 1), activation='relu', init='he_uniform')(pool_1)
            pool_2 = MaxPooling2D()(conv_2)
            conv_3 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', init='he_uniform')(pool_2)
            pool_3 = MaxPooling2D()(conv_3)

            flat = Flatten()(pool_3)

            hidden_1 = Dense(HIDDEN1_UNITS, activation='relu', init='he_uniform')(flat)
            hidden_2 = Dense(HIDDEN2_UNITS, activation='relu', init='he_uniform')(hidden_1)

            Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-5, name=name))(hidden_2)
            Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-5, name=name))(hidden_2)
            Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-6, name=name))(hidden_2)

            #V = Dense(action_dim, activation='tanh', init=init_output_layer)(hidden_2)
            V = merge([Steering,Acceleration,Brake],mode='concat')

            model = Model(input=S,output=V)
            self.summary_conv_3 = tf.image_summary('conv_3', conv_3, max_images=1)
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
