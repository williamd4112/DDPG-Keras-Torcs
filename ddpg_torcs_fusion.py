from __future__ import print_function
from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork_fusion import ActorNetwork
from CriticNetwork_fusion import CriticNetwork
from OU import OU
import timeit
import cv2

OU = OU()       #Ornstein-Uhlenbeck Process
import os
import sys

def playGame(checkpoints=None, train_indicator=1, eps=1.0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 40000
    BATCH_SIZE = 16
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.00075    #Learning rate for Actor
    LRC = 0.001    #Lerning rate for Critic

    vision = True
    action_dim = 3  #Steering/Acceleration/Brake

    if vision:
        state_vision_dim = (64, 64, 3)  #of sensors input
        state_sensor_dim = (3,)
    else:
        state_dim = 29
    np.random.seed(1337)

    EXPLORE = 100000.
    episode_count = 1000
    max_steps = 8000000
    reward = 0
    done = False
    step = 0
    epsilon = eps
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    summary_writer = tf.train.SummaryWriter('logs', graph_def=sess.graph_def)
    actor = ActorNetwork(sess, state_vision_dim, state_sensor_dim, action_dim, BATCH_SIZE, TAU, LRA, vision, summary_writer)
    critic = CriticNetwork(sess, state_vision_dim, state_sensor_dim, action_dim, BATCH_SIZE, TAU, LRC, vision)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=False, gear_change=False)
    log_file = open('train_log.log', 'w')
    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel_{}.h5".format(checkpoints))
        critic.model.load_weights("criticmodel_{}.h5".foramt(checkpoints))
        actor.target_model.load_weights("actormodel_{}.h5".format(checkpoints))
        critic.target_model.load_weights("criticmodel_{}.h5".format(checkpoints))
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")

    max_reward = 0
    min_reward = 0

    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        if vision:
            s_t = [ob.img, np.hstack((ob.angle, ob.speedX, ob.speedY))]
        else:
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0.
        total_damage = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            if vision:
                if buff.count() >= 0:
                    a_t_original = actor.model.predict([s_t[0].reshape((-1,) + state_vision_dim), s_t[1].reshape((-1,) + state_sensor_dim)])
                else:
                    a_t_original = a_t
            else:
                a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.25, 0.3)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.75 , 1.0, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])
            damage = ob.damage

            if vision:
                next_s_t = [ob.img, np.hstack((ob.angle, ob.speedX, ob.speedY))]
                if np.mod(step, 1) == 0:
                    buff.add(np.array(s_t), a_t[0], r_t, np.array(next_s_t), done)      #Add replay buffer
                s_t1 = next_s_t
            else:
                s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
                buff.add(s_t, a_t[0], r_t, s_t1, done)

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            states_vision = np.asarray([v[0] for v in states])
            states_sensor = np.asarray([s[1] for s in states])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            new_states_vision = np.asarray([v[0] for v in new_states])
            new_states_sensor = np.asarray([s[1] for s in new_states])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            if vision:
                new_states_vision_t = new_states_vision.reshape((-1,) + state_vision_dim)
                new_states_sensor_t = new_states_sensor.reshape((-1,) + state_sensor_dim)
                new_states_predict_t = actor.target_model.predict([new_states_vision_t, new_states_sensor_t]).reshape((-1,) + (action_dim,))
                target_q_values = critic.target_model.predict([new_states_vision_t, new_states_sensor_t, new_states_predict_t])
            else:
                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if train_indicator and buff.count() >= 1000:
                loss = critic.model.train_on_batch([states_vision, states_sensor, actions], y_t)
                a_for_grad = actor.model.predict([states_vision, states_sensor])
                grads = critic.gradients(states_vision, states_sensor, a_for_grad)
                actor.train(states_vision, states_sensor, grads)

                actor.target_train()
                critic.target_train()

            total_reward += r_t
            total_damage += damage
            s_t = s_t1

            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss, 'Buff' , buff.count())

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel_{}.h5".format(i), overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel_{}.h5".format(i), overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)
        max_reward = max(max_reward,total_reward)
        min_reward = min(min_reward,total_reward)
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward "+ str(total_reward) + "  EPS " + str(epsilon))
        print("Total Step: " + str(step) + ' Max: ' + str(max_reward) + ' Min: ' + str(min_reward))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    episode = int(sys.argv[1]) if len(sys.argv) >=2 else None
    eps = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0
    playGame(checkpoints=episode, eps=eps)
