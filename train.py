###########################################################################################
# Training of Deep Q-Learning Networks (DQN)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/Kchu/DeepRL_PyTorch
###########################################################################################
# The codes for the classical control game Pendulum are excerpted from OpenAI baseline Gym,
# and an new image data output API is added to this game by defining a new observation wrapper
# in wrappers.py.
# Parallel DQN for an Inverted Pendulum with Image Data implemented in PyTorch and OpenAI Gym
import os
import torch
import pickle
import time
import argparse
import numpy as np
from collections import deque

import dqn
from dqn import DQN
from wrappers import wrap, wrap_cover, wrap_cover_pendulun, wrap_cover_pendulun_test
from parallel import SubprocVecEnv
from pendulum import PendulumEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder


parser = argparse.ArgumentParser(description='Some settings of the experiment.')
parser.add_argument('--game', type=str, default='Pendulum', help='name of the games. for example: Breakout')

parser.add_argument('--idling', help=' ', action='store_true', default=False)
parser.add_argument('--n_envs', type=int, default=1)
parser.add_argument('--steps', type=int, default=150000)
parser.add_argument('--learn_start', type=int, default=1000)
parser.add_argument('--learn_freq', type=int, default=4)
parser.add_argument('--eps_init', type=float, default=1.0)
parser.add_argument('--eps_phase1_steps', type=int, default=50000)
parser.add_argument('--eps_phase2_steps', type=int, default=100000)
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--no_render', action='store_true')
parser.add_argument('--no_video', action='store_true')
parser.add_argument('--artifact_dir', type=str, default='data')

args = parser.parse_args()
if min(args.n_envs, args.steps, args.learn_freq, args.save_freq,
       args.eps_phase1_steps, args.eps_phase2_steps) < 1:
    parser.error('Environment count, steps, intervals and epsilon phases must be positive.')
if args.learn_start < 0 or not 0 <= args.eps_init <= 1:
    parser.error('learn_start must be nonnegative and eps_init must be in [0, 1].')
args.games = args.game

# the discrete action space
disc_actions = [np.array(-2, dtype=np.float32),
                np.array(-1.5, dtype=np.float32),
                np.array(-1, dtype=np.float32),
                np.array(-0.5, dtype=np.float32),
                np.array(0, dtype=np.float32),
                np.array(0.5, dtype=np.float32),
                np.array(1, dtype=np.float32),
                np.array(1.5, dtype=np.float32),
                np.array(2, dtype=np.float32)]

def main():
    '''Environment Settings'''
    # number of environments for C51
    N_ENVS = args.n_envs
    # Total simulation step
    STEP_NUM = args.steps
    # visualize for agent playing
    RENDERING = False
    # openai gym env name
    # ENV_NAME = args.game+'NoFrameskip-v4'
    ENV_NAME = args.game
    # idling
    IDLING = args.idling

    '''Training Settings'''
    # simulator steps before start learning, that is, storing transitions/samples into buffer first
    LEARN_START = args.learn_start
    # simulator steps for learning interval
    LEARN_FREQ = args.learn_freq
    # epsilon-greedy
    EPSILON = args.eps_init
    # the length pf each episode
    EPISODE_LENGTH = 200
    # create envs
    global disc_actions
    env = SubprocVecEnv([
        wrap_cover_pendulun(ENV_NAME, disc_actions, episode_length=EPISODE_LENGTH) for i in range(N_ENVS)])

    '''Save&Load Settings'''
    # check save/load
    SAVE = True
    LOAD = False
    BUFFER_LOAD = False
    # save frequency
    SAVE_FREQ = max(1, args.save_freq // N_ENVS)
    # paths for prediction net, target net, result log
    current_path = os.path.dirname(os.path.realpath(__file__))
    artifact_path = os.path.abspath(os.path.join(current_path, args.artifact_dir))
    PRED_PATH = os.path.join(artifact_path, 'model/dqn_pred_net_o_'+args.games+'.pkl')
    TARGET_PATH = os.path.join(artifact_path, 'model/dqn_target_net_o_'+args.games+'.pkl')
    RESULT_PATH = os.path.join(artifact_path, 'plots/dqn_result_o_'+args.games+'.pkl')
    BUFFER_PATH = os.path.join(artifact_path, 'replay_buffer/dqn_buffer_'+args.games+'.pkl')
    # create directory
    if not os.path.exists(os.path.dirname(RESULT_PATH)):
        os.makedirs(os.path.dirname(RESULT_PATH))
    if not os.path.exists(os.path.dirname(PRED_PATH)):
        os.makedirs(os.path.dirname(PRED_PATH))
    if not os.path.exists(os.path.dirname(TARGET_PATH)):
        os.makedirs(os.path.dirname(TARGET_PATH))
    if not os.path.exists(os.path.dirname(BUFFER_PATH)):
        os.makedirs(os.path.dirname(BUFFER_PATH))

    # define agent
    dqn = DQN()

    # model load with check
    if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
        dqn.load_model(PRED_PATH, TARGET_PATH)
        pkl_file = open(RESULT_PATH,'rb')
        result = pickle.load(pkl_file)
        pkl_file.close()
        print('Load complete!')
    else:
        result = []
        print('Initialize results!')

    # offline data load with check
    if BUFFER_LOAD and os.path.isfile(BUFFER_PATH):
        dqn.load_buffer(BUFFER_PATH)
        print('Load data complete!')

    print('Collecting experience...')

    # for storing the average returns of the latest 100 episodes
    epinfobuf = deque(maxlen=100)
    # check learning time
    start_time = time.time()

    # env reset
    s = env.reset()
    # print(s.shape)

    # for step in tqdm(range(1, STEP_NUM//N_ENVS+1)):
    for step in range(1, STEP_NUM // N_ENVS + 1):
        # When loading data from a file, don't need to sample from env.
        if not BUFFER_LOAD:
            a = dqn.choose_action(s, EPSILON, IDLING)

            # take action and get next state
            # s_, r, done, infos, _ = env.step(a)
            s_, r, done, infos, _ = env.step(a)
            if done.all():
                # log arrange
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo: epinfobuf.append(maybeepinfo)
                s_ = env.reset()

            # store the transition
            for i in range(N_ENVS):
                dqn.store_transition(s[i], a[i], r[i], s_[i], done[i])

        # annealing the epsilon(exploration strategy)
        used_steps = step * N_ENVS
        phase1_floor = min(0.1, args.eps_init)
        if used_steps <= args.eps_phase1_steps:
            EPSILON = args.eps_init - (args.eps_init - phase1_floor) * used_steps / args.eps_phase1_steps
        else:
            phase2_progress = min(1.0, (used_steps - args.eps_phase1_steps) / args.eps_phase2_steps)
            EPSILON = phase1_floor - (phase1_floor - min(0.001, phase1_floor)) * phase2_progress

        # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
        if (not IDLING) and (LEARN_START <= dqn.memory_counter) and (dqn.memory_counter % LEARN_FREQ == 0):
            loss = dqn.learn()

        # print log and save
        if step % SAVE_FREQ == 0:
            # check time interval
            time_interval = round(time.time() - start_time, 2)
            # calc mean return (the average return over the last 100 episodes)
            mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2) if epinfobuf else float('nan')
            result.append(mean_100_ep_return)
            # print log
            print('Used Step: ',dqn.memory_counter,
                '| EPS: ', round(EPSILON, 3),
                # '| Loss: ', loss,
                '| Mean ep 100 return: ', mean_100_ep_return,
                '| Used Time:',time_interval)
            # save model
            dqn.save_model(PRED_PATH, TARGET_PATH)
            # save the mean_100_ep_return (the average return over the last 100 episodes) to the binary file
            pkl_file = open(RESULT_PATH, 'wb')
            pickle.dump(np.array(result), pkl_file)
            pkl_file.close()
            # Render
            # Once the performance is good enough (>-200), deploy the learned greedy policy
            if mean_100_ep_return >= -200:  # trigger video only when performance is reasonable
                # Re-simulate to show the animation
                if not args.no_render:
                    evaluate_performance(dqn, disc_actions, episode_length=EPISODE_LENGTH)
                # re-simulate to save the video
                if not args.no_video:
                    save_animation(dqn, disc_actions)

                break

        if not BUFFER_LOAD:
            s = list(s_)
            if RENDERING:
                env.render()

    if not BUFFER_LOAD:
        dqn.save_buffer(BUFFER_PATH)
        env.close()
    print("The training is done!")

def evaluate_performance(agent, disc_actions, render_mode="human", episode_length=200, test_num=1):
    # create env
    global args
    game = args.game
    env = wrap_cover_pendulun_test(game, disc_actions, render_mode, episode_length)()

    for _ in range(test_num): # Play the game test_num times
        # initialize the env
        s, _ = env.reset()
        s = [s]
        # render
        env.render()

        for step in range(episode_length):
            a = agent.choose_action(s, 0.0, idling=False)
            s_, r, done, infos, _ = env.step(a[0])
            s = [s_]
            env.render()

    env.close()

def save_animation(agent, disc_actions, render_mode="rgb_array", frames_length=200):
    # create env
    # render_mode = "human" # for display the video in real-time
    # render_mode = "rgb_array" # for saving the video in files
    global args
    game = args.game
    env = wrap_cover_pendulun_test(game, disc_actions, render_mode, frames_length)()

    after_training = "after_training.mp4"
    after_video = VideoRecorder(env, after_training)
    # initialize the env
    s, _ = env.reset()
    s = [s]
    # render
    env.render()

    for step in range(frames_length):
        after_video.capture_frame()
        a = agent.choose_action(s, 0.0, idling=False)
        s_, r, done, infos, _ = env.step(a[0])
        s = [s_]
        env.render()

    after_video.close()
    env.close()

if __name__ == '__main__':
    main()
