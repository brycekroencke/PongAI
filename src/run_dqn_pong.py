from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer



def get_nonexistant_path(fname_path):

    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}-{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}-{}{}".format(filename, i, file_extension)
    return new_fname

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 1000000
batch_size = 32
gamma = 0.99
record_idx = 10000

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
model.load_state_dict(torch.load("model_pretrained.pth", map_location='cpu'))

target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_model.copy_from(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
if USE_CUDA:
    model = model.cuda()
    target_model = target_model.cuda()
    print("Using cuda")

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()

for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)

    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append((frame_idx, episode_reward))
        episode_reward = 0

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append((frame_idx, loss.data.cpu().numpy()))

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses, 0)[1]))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])




    if frame_idx % 50000 == 0:
        # torch.save(model.state_dict(), get_nonexistant_path("model.pth"))
        target_model.copy_from(model)

    if frame_idx % 100000 == 0:
        print("SAVING MODEL AND PLOTS")
        torch.save(model.state_dict(), get_nonexistant_path("model.pth"))

        r_plot_f = []
        r_plot_r = []
        l_plot_f = []
        l_plot_l = []
        for i in all_rewards:
            r_plot_f.append(i[0])
            r_plot_r.append(i[1])
        for i in losses:
            l_plot_f.append(i[0])
            l_plot_l.append(i[1])

        plt.plot(l_plot_f, l_plot_l)
        plt.ylabel('loss')
        plt.savefig(get_nonexistant_path("loss_plot"))
        plt.clf()

        plt.plot(r_plot_f, r_plot_r)
        plt.ylabel('reward')
        plt.savefig(get_nonexistant_path("reward_plot"))
        plt.clf()
        # plt.plot(losses[:][0], losses[:][1])
        # plt.ylabel('loss')
        # plt.savefig(get_nonexistant_path("loss_plot"))
        # plt.clf()
        #
        # plt.plot(all_rewards[:][0], all_rewards[:][1])
        # plt.ylabel('reward')
        # plt.savefig(get_nonexistant_path("reward_plot"))
        # plt.clf()
