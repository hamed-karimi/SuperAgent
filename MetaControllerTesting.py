import itertools
import math
from AgentExplorationFunctions import agent_reached_goal
# from Visualizer import Visualizer
import numpy as np
import torch
import matplotlib.pyplot as plt
from MetaController import MetaController
from copy import deepcopy
from matplotlib.ticker import FormatStrFormatter
from Agent import Agent
from ObjectFactory import ObjectFactory
from Environment import Environment
from Controller import Controller
from Utilities import Utilities


def get_predefined_parameters(num_object, param_name):
    if param_name == 'all_mental_states':
        all_param = [[-10, -5, 0, 5, 10]] * num_object
    elif param_name == 'all_object_rewards':
        all_param = [[0, 4, 8, 12, 16, 20]] * num_object
    elif param_name == 'all_mental_states_change':
        all_param = [[0, 1, 2, 3, 4, 5]] * num_object
    else:
        print('no such parameters')
        return
    num_param = len(all_param[0]) ** num_object
    param_batch = [] #torch.zeros((num_param, num_object))
    # ns = np.zeros((1, num_object))
    for i, ns in enumerate(itertools.product(*all_param)):
        # param_batch[i, :] = torch.tensor(ns)
        param_batch.append(list(ns))
    return param_batch



class MetaControllerVisualizer():
    def __init__(self, utility : Utilities):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_folder = utility.res_folder
        self.factory = ObjectFactory(utility)
        self.height = utility.params.HEIGHT
        self.width = utility.params.WIDTH
        self.object_type_num = utility.params.OBJECT_TYPE_NUM
        self.episode_num = utility.params.META_CONTROLLER_EPISODE_NUM
        self.all_actions = torch.tensor([[0, 0],
                                         [1, 0], [-1, 0], [0, 1], [0, -1],
                                         [1, 1], [-1, -1], [-1, 1], [1, -1]])
        self.action_mask = np.zeros((self.height, self.width, 1, len(self.all_actions)))
        self.initialize_action_masks()

        self.all_mental_states = get_predefined_parameters(self.object_type_num, 'all_mental_states')
        self.all_object_rewards = get_predefined_parameters(self.object_type_num, 'all_object_rewards')
        self.all_mental_states_change = get_predefined_parameters(self.object_type_num, 'all_mental_states_change')

        self.color_options = [[1, 0, .2], [0, .8, .2], [0, 0, 0]]
        self.goal_shape_options = ['*', 's', 'P', 'o', 'D', 'X']
        self.objects_color_name = ['red', 'green', 'black']  # 2: stay
        # self.row_num = 5
        # self.col_num = 6

    def initialize_action_masks(self):
        for i in range(self.height):
            for j in range(self.width):
                agent_location = torch.tensor([[i, j]])
                aa = np.ones((agent_location.size(0), len(self.all_actions)))
                for ind, location in enumerate(agent_location):
                    if location[0] == 0:
                        aa[ind, 2] = 0
                        aa[ind, 6] = 0
                        aa[ind, 7] = 0
                    if location[0] == self.height - 1:
                        aa[ind, 1] = 0
                        aa[ind, 5] = 0
                        aa[ind, 8] = 0
                    if location[1] == 0:
                        aa[ind, 4] = 0
                        aa[ind, 6] = 0
                        aa[ind, 8] = 0
                    if location[1] == self.width - 1:
                        aa[ind, 3] = 0
                        aa[ind, 5] = 0
                        aa[ind, 7] = 0
                self.action_mask[i, j, :, :] = aa

    def get_figure_title(self, mental_states):
        title = '$n_{0}: {1:.2f}'.format('{' + self.objects_color_name[0] + '}', mental_states[0, 0])
        for i in range(1, self.object_type_num):
            title += ", n_{0}: {1:.2f}$".format('{' + self.objects_color_name[i] + '}', mental_states[0, i])
        return title

    # def get_agent_goal_map(self, env_map, goal_location):
    #     agent_goal_map = torch.zeros_like(env_map[:, 1:, :, :])
    #     agent_goal_map[0, 0, :, :] = env_map[0, 0, :, :]
    #     agent_goal_map[0, 1, goal_location[0, 0], goal_location[0, 1]] = 1
    #     return agent_goal_map

    def get_object_shape_dictionary(self, environment: Environment):
        shape_map = dict()
        for obj_type in range(self.object_type_num):
            for at_obj in range(environment.each_type_object_num[obj_type]):
                key = tuple(environment.object_locations[obj_type, at_obj].tolist())
                shape_map[key] = self.goal_shape_options[at_obj]
        key = tuple(environment.agent_location[0].tolist())
        shape_map[key] = '.'
        return shape_map

    def next_agent_and_environment(self):
        for object_reward in self.all_object_rewards:
            episode_object_amount = [np.random.choice(['few', 'many']) for _ in range(self.object_type_num)]
            for mental_state_change in self.all_mental_states_change:
                for subplot_id, mental_state in enumerate(self.all_mental_states):
                    # pre_assigned_object_reward = [[]] * params.OBJECT_TYPE_NUM
                    pre_located_objects_location = [[[]]] * self.object_type_num
                    pre_located_objects_num = torch.zeros((self.object_type_num,), dtype=torch.int32)
                    prohibited_object_locations = []
                    # pre_located_agent = [[]]
                    # pre_assigned_mental_states = [[]]

                    for i in range(self.height):
                        for j in range(self.width):
                            test_agent = self.factory.get_agent(pre_location=[[i, j]],
                                                                preassigned_mental_states=[mental_state],
                                                                preassigned_mental_states_change=[mental_state_change])

                            test_environment = self.factory.get_environment(test_agent,
                                                                            episode_object_amount,
                                                                            [object_reward],
                                                                            pre_located_objects_num,
                                                                            pre_located_objects_location,
                                                                            prohibited_object_locations)
                            pre_located_objects_num = test_environment.each_type_object_num#.tolist()
                            pre_located_objects_location = test_environment.object_locations#.tolist()

                            yield test_environment, test_agent, subplot_id

    def get_goal_directed_actions(self, meta_controller: MetaController):
        meta_controller.policy_net.eval()
        fig, ax = None, None
        which_goal = None
        row_num = 5
        col_num = 5
        created_subplot = np.zeros((row_num*col_num, ), dtype=bool)
        for setting_id, outputs in enumerate(self.next_agent_and_environment()):
            environment = outputs[0]
            agent = outputs[1]
            subplot_id = outputs[2]

            # print('ms: ', agent.mental_states, 'or: ', environment.object_reward, 'sc: ', agent.states_change)
            if setting_id % (col_num * row_num * self.width * self.height) == 0:
                fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))
                # created_subplot[subplot_id] = True
            if setting_id % (self.height * self.width) == 0:
                which_goal = np.empty((self.height, self.width), dtype=str)
            # else:
            #     continue

            r = subplot_id // col_num
            c = subplot_id % col_num

            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            ax[r, c].invert_yaxis()

            shape_map = self.get_object_shape_dictionary(environment=environment) # not efficient doing this every time in the loop

            with torch.no_grad():
                goal_map, goal_location = meta_controller.get_goal_map(environment,
                                                                       agent,
                                                                       episode=0,
                                                                       epsilon=-1)

            which_goal[agent.location[0, 0], agent.location[0, 1]] = shape_map[tuple(goal_location.tolist())]
            goal_type = torch.where(environment.env_map[0, :, goal_location[0], goal_location[1]])[0].min()
            goal_type = 2 if goal_type == 0 else goal_type - 1
            selected_goal_shape = shape_map[tuple(goal_location.tolist())]
            size = 10 if selected_goal_shape == '.' else 50
            ax[r, c].scatter(agent.location[0, 1], agent.location[0, 0],
                             marker=selected_goal_shape,
                             s=size,
                             alpha=0.4,
                             facecolor=self.color_options[goal_type])

            if agent.location[0, 0] == self.height - 1 and agent.location[0, 1] == self.width - 1:
                # which_goal = np.empty((self.height, self.width), dtype=str)
                ax[r, c].set_title(self.get_figure_title(agent.mental_states), fontsize=10)

                for obj_type in range(self.object_type_num):
                    for obj in range(environment.object_locations.shape[1]):
                        if environment.object_locations[obj_type, obj, 0] == -1:
                            break
                        ax[r, c].scatter(environment.object_locations[obj_type, obj, 1],
                                         environment.object_locations[obj_type, obj, 0],
                                         marker=self.goal_shape_options[obj],
                                         s=200,
                                         edgecolor=self.color_options[obj_type],
                                         facecolor='none')
                ax[r, c].tick_params(length=0)
                ax[r, c].set(adjustable='box')
            if (setting_id+1) % (col_num * row_num * self.width * self.height) == 0:
                plt.tight_layout(pad=0.1, w_pad=6, h_pad=1)
                fig.savefig('{0}/or_{1}-{2}_msChange_{3}-{4}.png'.format(self.result_folder,
                                                                         environment.object_reward[0, 0],
                                                                         environment.object_reward[0, 1],
                                                                         agent.states_change[0, 0],
                                                                         agent.states_change[0, 1]))
                plt.close()

    def add_needs_plot(self, ax, agent_needs, global_index, r, c):
        ax[r, c].set_prop_cycle('color', self.color_options)
        ax[r, c].plot(agent_needs[:global_index, :], linewidth=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Needs', fontsize=9)
        return ax, r, c + 1

    def get_epsilon_plot(self, ax, r, c, steps_done, **kwargs):
        ax[r, c].scatter(np.arange(steps_done), kwargs['meta_controller_epsilon'], s=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Meta Controller Epsilon', fontsize=9)
        ax[r, c].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # ax[r, c].set_box_aspect(aspect=1)
        return ax, r, c + 1

    def policynet_values(self, environment, meta_controller):
        all_needs = torch.arange(-10, 11, 2)  # torch.tensor([-10, -5, 0, 5, 10])
        meta_controller.policy_net.eval()
        goal_type_num = environment.object_type_num + 1  # +1 for staying
        each_goal_type_num = environment.each_type_object_num + [1]
        for i in range(environment.height):
            for j in range(environment.width):
                row_num = goal_type_num
                col_num = max(max(each_goal_type_num), 2)
                fig, ax = plt.subplots(row_num, col_num, figsize=(22, 12))
                # shape_map = self.get_object_shape_dictionary(environment=environment)
                env_map = torch.zeros((1, 1 + self.object_type_num, self.height, self.width))  # +1 for agent layer
                env_map[0, 0, i, j] = 1
                env_map[0, 1:, :, :] = deepcopy(environment.env_map[0, 1:, :, :])
                # shape_map[(i, j)] = '.'  # Staying
                output_values = torch.zeros(all_needs.shape[0],
                                            all_needs.shape[0],
                                            environment.height,
                                            environment.width)
                with torch.no_grad():
                    for n1 in range(all_needs.shape[0]):
                        for n2 in range(all_needs.shape[0]):
                            need1 = all_needs[n1]
                            need2 = all_needs[n2]
                            need = torch.tensor([[need1, need2]])
                            output_values[n1, n2, :, :] = meta_controller.policy_net(env_map.to(self.device),
                                                                                     need.to(
                                                                                         self.device)).clone()  # 1 * 3
                            object_mask = env_map.sum(dim=1).squeeze()
                            output_values[n1, n2][object_mask < 1] = -math.inf
                r, c = -1, -1
                for goal_type in range(goal_type_num):
                    for goal in range(each_goal_type_num[goal_type]):
                        r = goal_type
                        c = goal
                        x = i if goal_type == goal_type_num - 1 else environment.object_locations[goal_type, goal, 0]
                        y = j if goal_type == goal_type_num - 1 else environment.object_locations[goal_type, goal, 1]
                        values = output_values[:, :, x, y]
                        rounded = [['%.2f' % elem for elem in row] for row in values.tolist()]
                        values_table = ax[r, c].table(cellText=rounded,
                                                      rowLabels=all_needs.tolist(),
                                                      colLabels=all_needs.tolist(),
                                                      colWidths=[.1 for i in range(values.shape[1])],
                                                      loc='center',
                                                      colLoc='right')
                        values_table.auto_set_font_size(False)
                        values_table.set_fontsize(6.5)
                        # values_table.scale(1, 2)
                        ax[r, c].set_title('agent: ({0}, {1}), object: {2} - ({3}, {4})'.format(i, j,
                                                                                                self.objects_color_name[
                                                                                                    goal_type], x, y),
                                           fontsize=10)
                        ax[r, c].axis('off')
                for goal_type in range(goal_type_num):
                    start_col = each_goal_type_num[goal_type] if goal_type != goal_type_num - 1 else 2
                    for empty_goal in range(start_col, col_num):
                        if empty_goal == col_num:
                            continue
                        ax[goal_type, empty_goal].axis('off')
                self.map_to_image(agent_location=torch.tensor([[i, j]]),
                                  environment=environment,
                                  ax=ax[r, c + 1])

                yield fig, ax, 'qvalues_agent_{0}_{1}'.format(i, j)
            # for i in range(self.height):
            #     for j in range(self.width):

    def add_needs_difference_hist(self, ax, agent_needs, needs_range, global_index, r, c):
        ax[r, c].set_prop_cycle('color', self.color_options)
        ax[r, c].hist(agent_needs[:global_index, 0] - agent_needs[:global_index, 1],
                      bins=np.linspace(needs_range[0] - needs_range[1], needs_range[1] - needs_range[0], 49),
                      linewidth=.1)
        ax[r, c].tick_params(axis='both', which='major', labelsize=9)
        ax[r, c].set_title('Needs', fontsize=9)
        return ax, r, c + 1

    def map_to_image(self, agent_location, environment, ax):

        # agent_location = environment.agent_location
        objects_location = environment.object_locations

        # fig, ax = plt.subplots(figsize=(15, 10))
        arrows_x = np.zeros((self.height, self.width))
        arrows_y = np.zeros((self.height, self.width))

        Xs = np.arange(0, self.height, 1)
        Ys = np.arange(0, self.width, 1)

        ax.quiver(Xs, Ys, arrows_x, arrows_y, scale=10)
        # ax.set_title("$n_{{red}}: {:.2f}, n_{{green}}: {:.2f}$".format(agent.need[0, 0], agent.need[0, 1]), fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        for obj_type in range(objects_location.shape[0]):
            for obj in range(environment.each_type_object_num[obj_type]):
                ax.scatter(objects_location[obj_type, obj, 1],
                           objects_location[obj_type, obj, 0],
                           marker='*', s=500, facecolor=self.color_options[obj_type])
        ax.set_box_aspect(aspect=1)

        ax.scatter(agent_location[0, 1], agent_location[0, 0], s=380, facecolors='b', edgecolors='k')
        plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)
        # return ax

    @staticmethod
    def get_qfunction_selected_goal_map(controller: Controller,
                                        meta_controller: MetaController,
                                        environment: Environment,
                                        agent: Agent):

        goal_map, goal_location = meta_controller.get_goal_map(environment,
                                                               agent,
                                                               episode=0,
                                                               epsilon=-1)  # get the goal map based on Q-values
        # rho = torch.tensor(0, dtype=torch.float)
        rho = []
        while True:
            agent_goal_map_0 = torch.stack([environment.env_map[:, 0, :, :], goal_map], dim=1)

            action_id = controller.get_action(agent_goal_map_0).clone()
            step_reward, dt = agent.take_action(environment, action_id)
            rho.append(step_reward)
            # rho.append(satisfaction - moving_cost - needs_cost)
            goal_reached = agent_reached_goal(environment, goal_map)

            if goal_reached:
                break

        return torch.tensor(rho).mean()
