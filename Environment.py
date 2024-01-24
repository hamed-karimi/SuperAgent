import random
import torch
import numpy as np
import warnings


class Environment:
    def __init__(self, few_many_objects, h, w, agent, num_object,
                 pre_assigned_object_reward,
                 pre_located_objects_num,
                 pre_located_objects_location,
                 prohibited_object_location):  # object_type_num is the number of each type of object
        self.each_type_object_num = None
        self.channels = 1 + num_object  # +1 is for agent which is the first layer
        self.height = h
        self.width = w
        self.object_type_num = num_object
        self.few_many_objects = few_many_objects
        self.agent_location = agent.location
        self.env_map = torch.zeros((1, self.channels, self.height, self.width),
                                   dtype=torch.float32)  # the 1 is for the env_map can be matched with the dimesions of weights (8, 2, 4, 4)
        self.object_locations = None
        self.object_reward = None
        self.object_reward_range = [0, 20]
        self.init_object_locations(pre_located_objects_num, pre_located_objects_location, prohibited_object_location)
        self.update_agent_location_on_map(agent)
        # self.reward_of_object = [reward_of_object] * agent.num_need
        self.init_object_reward(pre_assigned_object_reward)
        self.cost_of_staying = 0
        self.all_actions = torch.tensor([[0, 0],
                                         [1, 0], [-1, 0], [0, 1], [0, -1],
                                         [1, 1], [-1, -1], [-1, 1], [1, -1]])
        self.check_obj_need_compatibility(agent.num_mental_states)

    def check_obj_need_compatibility(self, num_agent_need):
        if self.object_type_num != num_agent_need:
            warnings.warn("The number of mental states and objects are not equal")

    def init_object_reward(self, pre_assigned_object_reward):
        if any(pre_assigned_object_reward):
            self.object_reward = torch.tensor(pre_assigned_object_reward)
        else:
            self.object_reward = torch.FloatTensor(1, self.object_type_num).uniform_(self.object_reward_range[0],
                                                                                     self.object_reward_range[1])

    def get_each_object_type_num_of_appearance(self):
        # e.g., self.few_many_objects : ['few', 'many']
        few_range = np.array([1, 2])
        many_range = np.array([3, 4])
        ranges = {'few': few_range,
                  'many': many_range}
        max_num = -1
        each_type_object_num = []
        for item in self.few_many_objects:
            at_type_obj_num = np.random.choice(ranges[item])
            each_type_object_num.append(at_type_obj_num)
            max_num = max(max_num, at_type_obj_num)
        object_locations = -1 * torch.ones(self.object_type_num, max_num, 2, dtype=torch.int32)
        return each_type_object_num, object_locations

    def init_objects_randomly(self, pre_located_objects_num, pre_located_objects_location,
                              prohibited_object_location):  # pre_located_objects is a list
        if any(pre_located_objects_num):  # some objects are pre-located
            self.each_type_object_num = pre_located_objects_num
            self.object_locations = -1 * torch.ones(self.object_type_num, max(pre_located_objects_num), 2,
                                                    dtype=torch.int32)
        else:
            self.each_type_object_num, self.object_locations = self.get_each_object_type_num_of_appearance()

        # pre-located objects
        for obj_type in range(self.object_type_num):
            for at_obj in range(pre_located_objects_num[obj_type]):
                if not torch.eq(pre_located_objects_location[obj_type, at_obj], torch.tensor([-1, -1])).all():
                    # temp = self.object_locations.clone()
                    self.object_locations[obj_type, at_obj, :] = torch.as_tensor(
                        pre_located_objects_location[obj_type][at_obj])
                    sample = self.object_locations[obj_type, at_obj, :]
                    self.env_map[0, 1 + obj_type, sample[0], sample[1]] = 1
        # New objects
        for obj_type in range(self.object_type_num):
            for at_obj in range(self.each_type_object_num[obj_type]):
                if torch.eq(self.object_locations[obj_type, at_obj], torch.tensor([-1, -1])).all():  # not pre-assigned
                    is_some_object_prohibited = len(prohibited_object_location) > 0
                    do = 1
                    while do:
                        hw_range = np.arange(self.height * self.width)
                        rand_num_in_range = random.choices(hw_range, k=1)[0]
                        sample = torch.tensor([rand_num_in_range // self.width, rand_num_in_range % self.width])
                        if is_some_object_prohibited:
                            if torch.eq(sample, prohibited_object_location).all():
                                continue
                        if sum(self.env_map[0, 1:, sample[0], sample[
                            1]]).item() == 0:  # This location is empty on every object layer as well as the pre-assigned objects
                            self.object_locations[obj_type, at_obj, :] = sample
                            self.env_map[0, 1 + obj_type, sample[0], sample[1]] = 1
                            # self.probability_map[rand_num_in_range] *= .9
                            do = 0

    def init_object_locations(self, pre_located_objects_num, pre_located_objects_location,
                              prohibited_object_location):  # Place objects on the map
        self.init_objects_randomly(pre_located_objects_num, pre_located_objects_location, prohibited_object_location)

    def update_agent_location_on_map(self, agent):
        # This is called by the agent (take_action method) after the action is taken
        self.env_map[0, 0, self.agent_location[0, 0], self.agent_location[0, 1]] = 0
        self.agent_location = agent.location.clone()
        self.env_map[0, 0, self.agent_location[0, 0], self.agent_location[0, 1]] = 1

    def get_reward(self):
        r = torch.zeros((1, self.object_type_num), dtype=torch.float32)
        goal_reached = torch.zeros(self.object_type_num, dtype=torch.float32)
        for obj in range(self.object_type_num):
            goal_reached[obj] = torch.all(torch.eq(self.agent_location[0], self.object_locations[obj, :, :]),
                                          dim=1).any().item()

            r[0, obj] += (goal_reached[obj] * self.object_reward[0, obj])
        return r, goal_reached

    def get_cost(self, action_id):
        if action_id == 0:
            return torch.tensor(self.cost_of_staying).float()
        return torch.linalg.norm(self.all_actions[action_id].float())
