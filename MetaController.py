import os.path
import pickle
import dill
from copy import deepcopy
import random
from torch import optim
import torch
from torch.optim.lr_scheduler import MultiplicativeLR
from DQN import hDQN, weights_init_orthogonal
from ReplayMemory import ReplayMemory
from collections import namedtuple
from torch import nn
import math
from os.path import join as pjoin
from os.path import exists as pexists


class MetaControllerMemory(ReplayMemory):
    def __init__(self, capacity, checkpoint_memory=None, checkpoint_weights=None, memory_size=0):
        super().__init__(capacity=capacity,
                         checkpoint_memory=checkpoint_memory,
                         checkpoint_weights=checkpoint_weights,
                         memory_size=memory_size)

    def get_transition(self, *args):
        Transition = namedtuple('Transition',
                                ('initial_map', 'initial_mental_states', 'goal_map', 'reward', 'n_steps', 'dt', 'final_map',
                                 'final_mental_states', 'agent_env_parameters'))
        return Transition(*args)


class MetaController:

    def __init__(self, params, pre_trained_weights_path=''):
        self.env_height = params.HEIGHT
        self.env_width = params.WIDTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = hDQN(params).to(self.device)
        if pre_trained_weights_path != "":
            self.policy_net.load_state_dict(torch.load(os.path.join(pre_trained_weights_path,
                                                                    'checkpoints',
                                                                    'policynet_checkpoint.pt'),
                                                       map_location=self.device))
        else:
            self.policy_net.apply(weights_init_orthogonal)
        self.target_net = hDQN(params).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.object_type_num = params.OBJECT_TYPE_NUM
        self.steps_done = 0
        self.EPS_START = 0.95
        self.EPS_END = 0.05
        self.episode_num = params.META_CONTROLLER_EPISODE_NUM
        self.episode_len = params.EPISODE_LEN
        self.target_net_update = params.META_CONTROLLER_TARGET_UPDATE
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=params.INIT_LEARNING_RATE)
        self.lr_scheduler = MultiplicativeLR(self.optimizer,
                                             lambda epoch: 1 / (1 + params.LEARNING_RATE_DECAY * epoch),
                                             last_epoch=-1, verbose=False)
        self.BATCH_SIZE = params.META_CONTROLLER_BATCH_SIZE
        self.gammas = []
        self.gamma_episodes = []
        self.gamma_delay_episodes = [0]
        self.gamma_max_delay = params.GAMMA_CASCADE_DELAY
        self.gamma_cascade = params.GAMMA_CASCADE
        self.max_gamma = params.MAX_GAMMA
        self.max_step_num = params.MAX_STEP_NUM
        self.min_gamma = 0
        self.all_gammas_ramped_up = False
        self.saved_target_nets = nn.ModuleList()
        # self.GAMMA = 0 if self.gamma_cascade else self.max_gamma
        self.batch_size_mul = 3
        # self.epsilon_list = []
        if params.CHECKPOINTS_DIR != "":
            self.initialize_from_checkpoint(params)
        else:
            self.memory = MetaControllerMemory(params.META_CONTROLLER_MEMORY_CAPACITY)

    def initialize_from_checkpoint(self, params):
        checkpoint_dir = params.CHECKPOINTS_DIR
        self.policy_net.load_state_dict(torch.load(pjoin(checkpoint_dir, 'policynet_checkpoint.pt')))
        self.target_net.load_state_dict(torch.load(pjoin(checkpoint_dir, 'targetnet_checkpoint.pt')))
        try:
            with open(pjoin(checkpoint_dir, 'meta_controller.pkl'), 'rb') as f1:
                checkpoint_dict = pickle.load(f1)
                self.gammas = checkpoint_dict['gammas']
                self.gamma_episodes = checkpoint_dict['gamma_episodes']
                self.gamma_delay_episodes = checkpoint_dict['gamma_delay_episode']
                self.all_gammas_ramped_up = checkpoint_dict['all_gammas_ramped_up']
            for q_i in range(len(self.gammas) - 1):
                Q = deepcopy(self.target_net)
                Q.load_state_dict(torch.load(pjoin(checkpoint_dir, 'Q_{0}_checkpoint.pt'.format(q_i))))
                self.saved_target_nets.append(deepcopy(Q))

            with open(pjoin(checkpoint_dir, 'memory.pkl'), 'rb') as f2:
                memory = dill.load(f2)
            with open(pjoin(checkpoint_dir, 'weights.pkl'), 'rb') as f3:
                weights = dill.load(f3)
            with open(pjoin(checkpoint_dir, 'train.pkl'), 'rb') as f4:
                train_dict = pickle.load(f4)
                last_episode = train_dict['episode']
        except:
            print('Memory, and Gammas not found. Continuing will empy memory')
            memory, weights = None, None
            last_episode = -1
        self.memory = MetaControllerMemory(params.META_CONTROLLER_MEMORY_CAPACITY,
                                           checkpoint_memory=memory,
                                           checkpoint_weights=weights,
                                           memory_size=last_episode + 1)

    def gamma_function(self, episode):
        m = 2
        ratio = m / self.target_net_update
        gamma = min(1 / (1 + math.exp(-episode * ratio + math.exp(2.3))),
                    self.max_gamma)
        return gamma

    def update_gammas(self):
        if self.gamma_cascade:
            if len(self.gammas) <= self.max_step_num:
                for g in range(len(self.gammas)):
                    self.gammas[g] = self.gamma_function(self.gamma_episodes[g])
                    self.gamma_episodes[g] += 1
            if ((len(self.gammas) > 0 and self.gammas[-1] == self.max_gamma) or (len(self.gammas) == 0)) \
                    and len(self.gammas) <= self.max_step_num and self.gamma_delay_episodes[-1] < self.gamma_max_delay:
                self.gamma_delay_episodes[-1] += 1

            elif ((0 < len(self.gammas) < self.max_step_num and self.gammas[-1] == self.max_gamma) or len(
                    self.gammas) == 0) and \
                    self.gamma_delay_episodes[-1] == self.gamma_max_delay:
                self.update_target_net()
                self.saved_target_nets.append(deepcopy(self.target_net))
                self.gammas.append(self.min_gamma)
                self.gamma_episodes.append(0)
                self.gamma_delay_episodes.append(0)
            if len(self.gammas) > 0 and self.gammas[-1] == self.max_gamma and \
                    len(self.gammas) == self.max_step_num and \
                    self.gamma_delay_episodes[-1] == self.gamma_max_delay:
                # all gammas ramped up completely, and we've waited for the delay
                self.all_gammas_ramped_up = True

    def get_nonlinear_epsilon(self, episode):
        x = math.log(episode + 1, self.episode_num)
        epsilon = -x ** 40 + 1
        return epsilon

    def get_linear_epsilon(self, episode):
        epsilon = self.EPS_START - (episode / self.episode_num) * \
                  (self.EPS_START - self.EPS_END)
        return epsilon

    def get_goal_map(self, environment, agent, episode, epsilon=None):
        # epsilon = self.get_nonlinear_epsilon(episode)
        goal_map = torch.zeros_like(environment.env_map[:, 0, :, :])
        if epsilon is None:
            epsilon = self.get_linear_epsilon(episode)

        # self.epsilon_list.append(epsilon)
        e = random.random()
        # all_object_locations = torch.stack(torch.where(environment.env_map[0, 1:, :, :]), dim=1)
        if e < epsilon:  # random (goal or stay)
            ######
            # stay_prob = .3
            # if random.random() <= stay_prob:  # Stay
            #     goal_location = environment.agent_location.squeeze()
            # else:  # Object
            #     goal_index = torch.randint(low=0, high=all_object_locations.shape[0], size=())
            #     goal_location = all_object_locations[goal_index, 1:]
            ######
            all_object_locations = torch.stack(torch.where(environment.env_map[0, :, :, :]), dim=1)
            goal_index = torch.randint(low=0, high=all_object_locations.shape[0], size=())
            goal_location = all_object_locations[goal_index, 1:]
        else:
            self.policy_net.eval()
            with torch.no_grad():
                env_map = environment.env_map.clone().to(self.device)
                mental_states = agent.mental_states.to(self.device)
                state_change = agent.states_change.to(self.device)
                object_reward = environment.object_reward.to(self.device)
                agent_env_params = torch.cat([state_change, object_reward], dim=1)
                output_values = self.policy_net(env_map, mental_states, agent_env_params)
                object_mask = environment.env_map.sum(dim=1)  # Either the agent or an object exists
                # object_mask = torch.ones_like(output_values)
                output_values[object_mask < 1] = -math.inf
                goal_location = torch.where(torch.eq(output_values, output_values.max()))
                goal_location = torch.as_tensor([ll[0] for ll in goal_location][1:])

        goal_map[0, goal_location[0], goal_location[1]] = 1
        self.steps_done += 1
        return goal_map, goal_location  # , goal_type

    def save_experience(self, initial_map, initial_mental_states, goal_map, acquired_reward, n_steps, dt, final_map, final_mental_states,
                        agent_env_parameters):
        self.memory.push_experience(initial_map, initial_mental_states, goal_map, acquired_reward, n_steps, dt, final_map,
                                    final_mental_states, agent_env_parameters)
        # memory_prob = 1
        # self.memory.push_selection_ratio(selection_ratio=memory_prob)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self):
        if self.memory.__len__() < self.BATCH_SIZE * self.batch_size_mul:
            return float('nan')
        transition_sample = self.memory.sample(self.BATCH_SIZE)
        batch = self.memory.get_transition(*zip(*transition_sample))
        self.policy_net.train()

        initial_map_batch = torch.cat([batch.initial_map[i] for i in range(len(batch.initial_map))]).to(self.device)
        initial_need_batch = torch.cat([batch.initial_mental_states[i] for i in range(len(batch.initial_mental_states))]).to(self.device)
        goal_map_batch = torch.cat(batch.goal_map).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        n_steps_batch = torch.cat(batch.n_steps).to(self.device)
        final_map_batch = torch.cat([batch.final_map[i] for i in range(len(batch.final_map))]).to(self.device)
        final_need_batch = torch.cat([batch.final_mental_states[i] for i in range(len(batch.final_mental_states))]).to(self.device)
        final_map_object_mask_batch = final_map_batch.sum(dim=1)
        agent_env_parameters = torch.cat([batch.agent_env_parameters[i] for i in range(len(batch.agent_env_parameters))]).to(self.device)

        policynet_goal_values_of_initial_state = self.policy_net(initial_map_batch,
                                                                 initial_need_batch,
                                                                 agent_env_parameters).to(self.device)

        goal_values_of_selected_goals = policynet_goal_values_of_initial_state[goal_map_batch == 1]

        steps_discounts = torch.zeros(reward_batch.shape,
                                      device=self.device)
        steps_discounts[:, :len(self.gammas)] = torch.as_tensor(self.gammas, device=self.device)
        steps_discounts = torch.cat([torch.ones(steps_discounts.shape[0], 1, device=self.device),
                                     steps_discounts], dim=1)  # step reward is not discounted

        cum_steps_discounts = torch.cumprod(steps_discounts, dim=1)[:, :-1]  # step reward is not discounted, so the
        # num of gammas for discounting rewards is
        # one less than for Q
        discounted_reward = (reward_batch * cum_steps_discounts).sum(dim=1)

        q_gammas = torch.cumprod(steps_discounts[:, 1:], dim=1).gather(dim=1,
                                                                       index=n_steps_batch.unsqueeze(dim=1).long() - 1)
        # q_gammas = torch.cumprod(steps_discounts, dim=1).gather(dim=1,
        #                                                         index=n_steps_batch.unsqueeze(dim=1).long())
        if not self.all_gammas_ramped_up:
            targetnet_goal_values_of_final_state = torch.zeros_like(policynet_goal_values_of_initial_state)
            # all_targetnets = deepcopy(self.saved_target_nets)
            # all_targetnets.append(self.target_net)
            outlook = len(self.gammas) + 1
            remaining_steps = outlook - n_steps_batch

            # max number of gammas should be 6 not 7. We take at most 7 steps, so we should look forward to n_gamms+1, thus making n_gammas 6.

            have_remaining_steps = remaining_steps > 0
            which_q = -1 * torch.ones_like(have_remaining_steps, dtype=torch.int32)
            which_q[have_remaining_steps] = remaining_steps[have_remaining_steps] - 1
            for q_i in range(len(self.saved_target_nets)):
                use_q_i = (which_q == q_i)
                targetnet_goal_values_of_final_state[use_q_i, :, :] = self.saved_target_nets[q_i](final_map_batch[use_q_i],
                                                                                                  final_need_batch[use_q_i],
                                                                                                  agent_env_parameters[use_q_i])

        else:
            targetnet_goal_values_of_final_state = self.target_net(final_map_batch,
                                                                   final_need_batch,
                                                                   agent_env_parameters).to(self.device)

        targetnet_goal_values_of_final_state[final_map_object_mask_batch < 1] = -math.inf
        targetnet_max_goal_value = torch.amax(targetnet_goal_values_of_final_state,
                                              dim=(1, 2)).detach().float()

        expected_goal_values = discounted_reward + targetnet_max_goal_value * q_gammas.squeeze()

        criterion = nn.SmoothL1Loss()
        loss = criterion(goal_values_of_selected_goals, expected_goal_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.update_gammas()
        return loss
