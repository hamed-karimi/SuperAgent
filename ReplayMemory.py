import math
from collections import deque
import random
from datetime import datetime
import numpy as np
import torch


class ReplayMemory():
    def __init__(self, capacity, checkpoint_memory=None, checkpoint_weights=None, memory_size=0):
        # We keep track of the number of experiences,
        # and we overwrite the early experiences by the later,
        # after episode_num reaches max capacity
        self.max_len = capacity
        if checkpoint_memory is None:
            self.experience_index = 0
            # self.weights_size = 0
            self.memory = np.zeros((self.max_len, ), dtype=object)
            # self.weights = np.zeros((self.max_len, ), dtype=np.float)
            # self.weights_exp_sum = 0
            # self.weights_exp = np.ones((self.max_len, ))
        else:
            self.memory = checkpoint_memory
            # self.weights = checkpoint_weights
            self.experience_index = memory_size
            # self.weights_size = experience_index
            # self.weights_exp_sum = sum(np.exp(self.weights[:self.experience_index]))
            # self.weights_exp = np.exp(self.weights[:self.experience_index])
        print('memory size: ', self.experience_index)

    def get_transition(self, *args):
        pass

    def push_experience(self, *args):
        self.memory[self.experience_index % self.max_len] = self.get_transition(*args)
        self.experience_index += 1
        # if self.experience_index == self.max_len:
        #     self.memory[:-1] = self.memory[1:]
        #     self.memory[-1] = self.get_transition(*args)
        # else:
        #     self.memory[self.experience_index] = self.get_transition(*args)
        #     self.experience_index += 1

    # def push_selection_ratio(self, **kwargs):
    #     if self.weights_size == self.max_len:
    #         self.weights_exp_sum = self.weights_exp_sum - np.exp(self.weights[0]) + np.exp(kwargs['selection_ratio'])
    #
    #         self.weights_exp[:-1] = self.weights_exp[1:]
    #         self.weights_exp[-1] = np.exp(kwargs['selection_ratio'])
    #
    #         self.weights[:-1] = self.weights[1:]
    #         self.weights[-1] = kwargs['selection_ratio']
    #     else:
    #         self.weights_exp_sum += np.exp(kwargs['selection_ratio'])
    #         self.weights_exp[self.weights_size] = np.exp(kwargs['selection_ratio'])
    #         self.weights[self.weights_size] = kwargs['selection_ratio']
    #         self.weights_size += 1

    def weighted_sample_without_replacement(self, k):
        sample_indices = random.sample(range(0, min(self.experience_index, self.max_len)), k)
        sample = self.memory[sample_indices]
        # sample = np.random.choice(self.memory[:self.experience_index],
        #                           size=k,
        #                           replace=False)

        # sample = np.random.choice(self.memory[:self.experience_index],
        #                           size=k,
        #                           replace=False,
        #                           p=self.weights_exp[:self.weights_size] / self.weights_exp_sum)
        # np.exp(self.weights[:self.experience_index])
        # sum(np.exp(self.weights[:self.experience_index]))
        return sample
        # v = [rng.random() ** (1 / w) for w in self.weights]
        # order = sorted(range(len(self.memory)), key=lambda i: v[i])
        # return [self.memory[i] for i in order[-k:]]

    def sample(self, size):
        # return random.choices(self.memory, weights=self.weights, k=size)
        sample = self.weighted_sample_without_replacement(k=size)
        return sample



        # first_steps_sample_size = math.floor(size * self.first_steps_sample_ratio)
        # other_steps_sample_size = size - first_steps_sample_size
        #
        # first_sample_indices = torch.multinomial(torch.ones(math.floor(len(self.early_memory)*0.1), dtype=torch.float),
        #                                          num_samples=first_steps_sample_size,
        #                                          replacement=False)
        # first_samples = [self.early_memory[i] for i in first_sample_indices]
        #
        # weights = torch.cat([torch.ones_like(torch.as_tensor(self.early_memory_weights, dtype=torch.float)),
        #                      torch.ones_like(torch.as_tensor(self.later_memory_weights, dtype=torch.float))])
        #
        # weights[first_sample_indices] = 0
        # sample_indices = torch.multinomial(weights,
        #                                    num_samples=other_steps_sample_size,
        #                                    replacement=False)
        # early_memory_indices = sample_indices[sample_indices < len(self.early_memory)]
        # later_memory_indices = sample_indices[sample_indices >= len(self.early_memory)] - len(self.early_memory)
        #
        # early_samples = [self.early_memory[i] for i in early_memory_indices]
        # later_samples = [self.later_memory[i] for i in later_memory_indices]
        #
        # return first_samples+early_samples+later_samples

    def __len__(self):
        return self.experience_index % self.max_len
        # return len(self.memory)
        # return len(self.early_memory) + len(self.later_memory)
