###########################################################################################
# Implementation of Deep Q-Learning Networks (DQN)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/Kchu/DeepRL_PyTorch
###########################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from replay_memory import ReplayBufferImage


'''DQN settings'''
# sequential images to define state
STATE_LEN = 4
# target policy sync interval
TARGET_REPLACE_ITER = 2
# (prioritized) experience replay memory size
MEMORY_CAPACITY = int(1e+5)
# gamma for MDP
GAMMA = 0.99

'''Training settings'''
# check GPU usage
USE_GPU = torch.cuda.is_available()
print('USE GPU: '+str(USE_GPU))
# mini-batch size
BATCH_SIZE = 64
LR = 2e-4
N_ACTIONS = 9
N_STATE = 4
N_TILE = 20
if 'DQN_BATCH' in os.environ:
    BATCH_SIZE = int(os.environ['DQN_BATCH'])
if 'DQN_LR' in os.environ:
    LR = float(os.environ['DQN_LR'])
if 'DQN_TARGET_REPLACE_ITER' in os.environ:
    TARGET_REPLACE_ITER = int(os.environ['DQN_TARGET_REPLACE_ITER'])
if 'DQN_N_TILE' in os.environ:
    N_TILE = int(os.environ['DQN_N_TILE'])
SOFT_UPDATE_RATE = float(os.environ.get('DQN_SOFT_UPDATE_RATE', '0.01'))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.feature_extraction = nn.Sequential(
            # Conv2d(input channels, output channels, kernel_size, stride)
            nn.Conv2d(STATE_LEN, 8, kernel_size=8, stride=2),

            # TODO: ADD SUITABLE CNN LAYERS TO ACHIEVE BETTER PERFORMANCE
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # action value
        self.fc_q = nn.Linear(8 * 8 * 8 + N_TILE * N_STATE, N_ACTIONS)

        # TODO: ADD SUITABLE FULLY CONNECTED LAYERS TO ACHIEVE BETTER PERFORMANCE
        self.fc_0 = nn.Linear(6 * 6 * 8 + N_TILE * N_STATE, 256)
        self.fc_1 = nn.Linear(256, 8 * 8 * 8 + N_TILE * N_STATE)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, state):
        # x.size(0) : minibatch size
        mb_size = x.size(0)
        x = self.feature_extraction(x / 255.0) # (m, 9 * 9 * 10)
        x = x.view(x.size(0), -1)
        state = state.view(state.size(0), -1)
        state = torch.tile(state, (1, N_TILE))
        x = torch.cat((x, state), 1)
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        action_value = self.fc_q(x)

        return action_value

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))


class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync target net
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()

        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        # loss function
        self.loss_function = nn.MSELoss()
        # create the replay buffer
        self.replay_buffer = ReplayBufferImage(MEMORY_CAPACITY)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)

    def update_target(self, target, pred, update_rate):
        # update target network parameters using prediction network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate * pred_param.data)

    def save_model(self, pred_path, target_path):
        # save prediction network and target network
        self.pred_net.save(pred_path)
        self.target_net.save(target_path)

    def load_model(self, pred_path, target_path):
        # load prediction network and target network
        self.pred_net.load(pred_path)
        self.target_net.load(target_path)

    def save_buffer(self, buffer_path):
        self.replay_buffer.save_data(buffer_path)
        print("Successfully save buffer!")

    def load_buffer(self, buffer_path):
        # load data from the pkl file
        self.replay_buffer.read_list(buffer_path)

    def choose_action(self, s, epsilon, idling):

        # TODO: REPLACE THE FOLLOWING PLACEHOLDER CODE WITH YOUR CODE

        # NOTE: If you enabled multiple threading (N_ENVS number of environments), then you should sample N_ENVS number of
        # actions that matches the number of environments. E.g.,
        # action = np.random.randint(0, N_ACTIONS, N_ENVS)
        # where N_ENVS can be inferred from the appropriate dimensions of s (which stacks the states from all N_ENVS env's).
        # For example:
        image = np.stack([item[0] for item in s])
        state = np.stack([item[1] for item in s])
        N_ENVS = image.shape[0]

        # Epsilon-greedy policy
        if np.random.uniform() < epsilon:
            # Exploration: random action
            action = np.random.randint(0, N_ACTIONS, N_ENVS)
        else:
            # Exploitation: choose best action based on Q-values
            with torch.no_grad():
                image_tensor = torch.FloatTensor(image)
                state_tensor = torch.FloatTensor(state)

                if USE_GPU:
                    image_tensor = image_tensor.cuda()
                    state_tensor = state_tensor.cuda()

                # Get Q-values for all actions
                q_values = self.pred_net(image_tensor, state_tensor)
                # Select action with maximum Q-value
                action = q_values.argmax(dim=1).cpu().numpy()

        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):

        # TODO: REPLACE THE FOLLOWING PLACEHOLDER CODE WITH YOUR CODE

        # Update target network periodically
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, SOFT_UPDATE_RATE)
        self.learn_step_counter += 1

        # Sample a minibatch from replay buffer
        batch = self.replay_buffer.sample(BATCH_SIZE)
        b_i, b_s, b_a, b_r, b_pi, b_ps, b_d = batch
        batch_image = torch.FloatTensor(b_i)
        batch_state = torch.FloatTensor(b_s)
        batch_action = torch.LongTensor(b_a).unsqueeze(1)
        batch_reward = torch.FloatTensor(b_r).unsqueeze(1)
        batch_image_next = torch.FloatTensor(b_pi)
        batch_state_next = torch.FloatTensor(b_ps)
        batch_done = torch.FloatTensor(b_d).unsqueeze(1)

        # Move to GPU if available
        if USE_GPU:
            batch_image = batch_image.cuda()
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_image_next = batch_image_next.cuda()
            batch_state_next = batch_state_next.cuda()
            batch_done = batch_done.cuda()

        # Compute Q(s, a) - current Q values
        q_eval = self.pred_net(batch_image, batch_state).gather(1, batch_action)

        # Compute target Q values: r + gamma * max_a' Q(s', a')
        with torch.no_grad():
            q_next = self.target_net(batch_image_next, batch_state_next).max(1, keepdim=True)[0]
            q_target = batch_reward + GAMMA * q_next * (1 - batch_done)

        # Compute loss
        loss = self.loss_function(q_eval, q_target)

        # Optimize the prediction network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
