import numpy as np
import torch
from torch.nn import ReLU, Sigmoid
import random
from copy import deepcopy
from itertools import product


class Agent:
    def __init__(self, h, w, n, prob_init_needs_equal, predefined_location,
                 preassigned_mental_states, preassigned_state_change, lambda_satisfaction, epsilon_function='Linear'):  # n: number of all_mental_states
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = h
        self.width = w
        self.location = self.initial_location(predefined_location)
        self.num_mental_states = n
        self.range_of_initial_states = [-12, 12]
        self.range_of_state_change = [0, 5]
        self.prob_init_needs_equal = prob_init_needs_equal
        self.mental_states = None
        self.states_change = None
        self.set_mental_states(preassigned_mental_states)
        self.set_mental_state_change(preassigned_state_change)
        self.steps_done = 0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        # self.b_matrix = need_increase  # How much the all_mental_states increases after each action
        self.lambda_satisfaction = lambda_satisfaction
        self.lambda_cost = 1
        self.no_reward_threshold = -5
        self.relu = ReLU()
        self.rho_function = ReLU()
        possible_h_w = [list(range(h)), list(range(w))]
        self.epsilon_function = epsilon_function
        # self.all_locations = torch.from_numpy(np.array([element for element in product(*possible_h_w)]))

    def poly_relu(self, x, p=2):
        return self.relu(x) ** p

    def set_mental_state_change(self, preassigned_state_change):
        if any(preassigned_state_change):
            self.states_change = torch.tensor(preassigned_state_change)
        else:
            self.states_change = torch.FloatTensor(1, self.num_mental_states).uniform_(self.range_of_state_change[0],
                                                                                       self.range_of_state_change[1])

    def set_mental_states(self, preassigned_states):
        if any(preassigned_states):
            mental_states = torch.tensor(preassigned_states)
        else:
            p = random.uniform(0, 1)
            if p <= self.prob_init_needs_equal:
                mental_states = torch.rand((1, self.num_mental_states))
                mental_states[0, 1:] = mental_states[0, 0]
            else:
            #     all_mental_states = torch.rand((1, self.num_need))
            # all_mental_states = (self.range_of_initial_states[1] - self.range_of_initial_states[0]) * all_mental_states + \
            #                 self.range_of_initial_states[0]
                mental_states = torch.FloatTensor(1, self.num_mental_states).uniform_(self.range_of_initial_states[0],
                                                                                      self.range_of_initial_states[1])
        self.mental_states = mental_states

    def initial_location(self, predefined_location):  # predefined_location is a list
        if len(predefined_location[0]) > 0:
            return torch.tensor(predefined_location)
        return torch.from_numpy(np.asarray((np.random.randint(self.height), np.random.randint(self.width)))).unsqueeze(
            0)

    # def reward_function(self, need):
    #     x = need.clone()
    #     pos = self.relu(x)
    #     neg = x
    #     neg[x > 0] = 0
    #     neg[x < self.no_reward_threshold] = self.no_reward_threshold  # (pow(1.1, self.no_reward_threshold) - 1) * 7
    #     return pos + neg

    def update_mental_state(self, dt):
        dz = (self.states_change * dt)
        self.mental_states += dz

    def update_mental_states_after_object(self, u):  # u > 0
        # adjusted_u = self.reward_function(self.all_mental_states) - self.reward_function(self.all_mental_states - u)
        # self.all_mental_states = self.all_mental_states - adjusted_u
        self.mental_states += -(1 * u)
        self.mental_states = torch.maximum(self.mental_states, torch.tensor(self.no_reward_threshold))

    # def update_need_after_step(self, time_past):
    #     for i in range(self.num_need):
    #         self.all_mental_states[0, i] += (self.lambda_need * time_past)
    #
    # def update_need_after_reward(self, reward):
    #     adjusted_reward = self.reward_function(self.all_mental_states) - self.reward_function(self.all_mental_states - reward)
    #     self.all_mental_states = self.all_mental_states - adjusted_reward

    def total_positive_need(self):
        total_need = self.rho_function(self.mental_states).sum().squeeze()
        return total_need

    def take_action(self, environment, action_id):
        # print('environment.all_actions: ', environment.all_actions.device)
        selected_action = environment.all_actions[action_id].squeeze()  # to device
        self.location[0, :] += selected_action
        step_length = environment.get_cost(action_id)
        time_passed = torch.tensor(1.) if step_length < 1.4 else step_length
        carried_total_need = self.total_positive_need()

        moving_cost = self.lambda_cost * step_length
        needs_cost = time_passed * carried_total_need

        environment.update_agent_location_on_map(self)
        object_reward, _ = environment.get_reward()
        # self.update_need_after_step(time_passed)
        # last_total_need = self.get_total_need()

        # self.update_need_after_reward(f)
        self.update_mental_state(dt=time_passed)
        positive_need_before_object = self.total_positive_need()
        self.update_mental_states_after_object(u=object_reward)
        positive_need_after_object = self.total_positive_need()
        mental_states_reward = self.relu(positive_need_before_object - positive_need_after_object) * self.lambda_satisfaction
        step_reward = mental_states_reward - moving_cost - needs_cost
        return step_reward, time_passed
