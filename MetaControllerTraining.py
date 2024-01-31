import os.path
from os.path import exists as pexists
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pickle
import torch
import dill
from torch.utils.tensorboard import SummaryWriter
from MetaControllerTesting import MetaControllerVisualizer
from ObjectFactory import ObjectFactory
from AgentExplorationFunctions import *


def training_meta_controller(utility):
    params = utility.params
    res_folder = utility.make_res_folder(sub_folder='MetaController')
    start_episode = utility.get_last_episode() + 1
    print('start episode: ', start_episode)
    writer = SummaryWriter()
    factory = ObjectFactory(utility)
    controller = factory.get_controller()
    meta_controller = factory.get_meta_controller()
    # meta_controller_visualizer = MetaControllerVisualizer(utility)
    # environment_initialization_prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)
    print((torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    for episode in range(start_episode, params.META_CONTROLLER_EPISODE_NUM):
        episode_begin = True
        episode_q_function_selected_goal_reward = 0
        episode_meta_controller_reward = 0
        episode_meta_controller_loss = 0
        # all_actions = 0
        pre_assigned_object_reward = [[]] * params.OBJECT_TYPE_NUM
        pre_located_objects_location = [[[]]] * params.OBJECT_TYPE_NUM
        prohibited_object_locations = []
        pre_located_objects_num = torch.zeros((params.OBJECT_TYPE_NUM,), dtype=torch.int32)
        pre_located_agent = [[]]
        pre_assigned_mental_states = [[]]
        pre_assigned_mental_states_change = [[]]
        object_amount_options = ['few', 'many']
        episode_object_amount = [np.random.choice(object_amount_options) for _ in range(params.OBJECT_TYPE_NUM)]
        # Initialized later in the reached if-statement
        environment = None
        agent = None

        for goal_selecting_step in range(params.EPISODE_LEN):
            steps = 0
            # steps_rho = []
            steps_reward = 0
            action_rewards = []
            action_rewards_tensor = torch.empty(1)
            # step_moving_costs = []
            # step_needs_costs = []
            step_dt = []
            if episode_begin:
                agent = factory.get_agent(pre_located_agent, pre_assigned_mental_states, pre_assigned_mental_states_change)
                environment = factory.get_environment(agent,
                                                      episode_object_amount,
                                                      pre_assigned_object_reward=pre_assigned_object_reward,
                                                      pre_located_objects_num=pre_located_objects_num,
                                                      pre_located_objects_location=pre_located_objects_location,
                                                      prohibited_object_location=prohibited_object_locations)
                episode_begin = False

            environment_0 = deepcopy(environment)
            agent_0 = deepcopy(agent)
            goal_map, goal_location = meta_controller.get_goal_map(environment,
                                                                   agent,
                                                                   episode)
            # Take action
            while True:
                # env_map_0 = environment.env_map.clone()
                # need_0 = agent.need.clone()
                agent_goal_map_0 = torch.stack([environment.env_map[:, 0, :, :], goal_map], dim=1)
                # print('agent_goal_map_0: ', agent_goal_map_0.device)
                action_id = controller.get_action(agent_goal_map_0).clone()
                # print('action_id: ', action_id.device)
                action_reward, dt = agent.take_action(environment, action_id)
                action_rewards.append(action_reward)
                # step_moving_costs.append(moving_cost)
                # step_needs_costs.append(needs_cost)
                step_dt.append(dt)
                # steps_rho.append(rho)

                goal_reached = agent_reached_goal(environment, goal_map)

                steps += 1
                if goal_reached:
                    pre_located_objects_location, prohibited_object_locations = update_pre_located_objects(
                        environment.object_locations,
                        agent.location,
                        goal_reached)

                    pre_assigned_object_reward = environment.object_reward.tolist()
                    pre_located_objects_num = environment.each_type_object_num
                    pre_located_agent = agent.location.tolist()
                    pre_assigned_mental_states = agent.mental_states.tolist()
                    pre_assigned_mental_states_change = agent.states_change.tolist()

                    agent = factory.get_agent(pre_located_agent, pre_assigned_mental_states, pre_assigned_mental_states_change)

                    environment = factory.get_environment(agent,
                                                          episode_object_amount,
                                                          pre_assigned_object_reward,
                                                          pre_located_objects_num,
                                                          pre_located_objects_location,
                                                          prohibited_object_locations)

                    action_rewards_tensor = torch.tensor(action_rewards)
                    # moving_cost_tensor = torch.tensor(step_moving_costs)
                    # needs_cost_tensor = torch.tensor(step_needs_costs)
                    dt_tensor = torch.tensor(step_dt).unsqueeze(dim=0).sum(dim=1)
                    steps_tensor = torch.tensor([steps], dtype=torch.int32)
                    agent_env_parameters = torch.cat([agent.states_change, environment.object_reward], dim=1)
                    # steps_reward = torch.zeros(1, params.HEIGHT + params.WIDTH - 2)
                    steps_reward = torch.zeros(1, max(params.HEIGHT-1, params.WIDTH-1))
                    steps_reward[0, :steps] = action_rewards_tensor.unsqueeze(dim=0)

                    meta_controller.save_experience(environment_0.env_map, agent_0.mental_states, goal_map,
                                                    steps_reward,
                                                    steps_tensor,
                                                    dt_tensor, environment.env_map.clone(), agent.mental_states.clone(),
                                                    agent_env_parameters)

                if goal_reached or steps == params.EPISODE_STEPS:
                    break

            episode_meta_controller_reward += steps_reward[0, :steps].mean()
            at_loss = meta_controller.optimize()
            episode_meta_controller_loss += get_meta_controller_loss(at_loss)
            episode_q_function_selected_goal_reward += MetaControllerVisualizer.get_qfunction_selected_goal_map(
                controller,
                meta_controller,
                deepcopy(environment_0),
                deepcopy(agent_0))

        if episode_meta_controller_loss > 0:
            meta_controller.lr_scheduler.step()
        writer.add_scalar("Meta Controller/Loss", episode_meta_controller_loss / params.EPISODE_LEN, episode)
        writer.add_scalar("Meta Controller/Reward", episode_meta_controller_reward / params.EPISODE_LEN, episode)
        gamma_dict = {}
        for g in range(len(meta_controller.gammas)):
            gamma_dict['gamma_{0}'.format(g)] = meta_controller.gammas[g]

        writer.add_scalars(f'Meta Controller/Gamma', gamma_dict, episode)
        writer.add_scalar('Meta Controller/Q-values goal reward',
                          episode_q_function_selected_goal_reward / params.EPISODE_LEN,
                          episode)
        # if (episode + 1) % params.PRINT_OUTPUT == 0:
        #     pre_assigned_object_reward = [[]] * params.OBJECT_TYPE_NUM
        #     pre_located_objects_location = [[[]]] * params.OBJECT_TYPE_NUM
        #     pre_located_objects_num = torch.zeros((params.OBJECT_TYPE_NUM,), dtype=torch.int32)
        #     prohibited_object_locations = []
        #     pre_located_agent = [[]]
        #     pre_assigned_mental_states = [[]]
        #     test_agent = factory.get_agent(pre_located_agent, pre_assigned_mental_states)
        #     test_environment = factory.get_environment(test_agent,
        #                                                episode_object_amount,
        #                                                pre_assigned_object_reward,
        #                                                pre_located_objects_num,
        #                                                pre_located_objects_location,
        #                                                prohibited_object_locations)
        #
        #     fig, ax = meta_controller_visualizer.get_goal_directed_actions(test_environment,
        #                                                                    meta_controller,
        #                                                                    controller)
        #     fig.savefig('{0}/episode_{1}.png'.format(res_folder, episode + 1))
        #     plt.close()

        if (episode + 1) % params.META_CONTROLLER_TARGET_UPDATE == 0:
            meta_controller.update_target_net()
            print('META CONTROLLER TARGET NET UPDATED')
        if (episode + 1) % params.CHECKPOINT_SAVE_FREQUENCY == 0:
            checkpoints_dir = pjoin(res_folder, 'checkpoints')
            if not pexists(checkpoints_dir):
                os.mkdir(checkpoints_dir)
            torch.save(meta_controller.policy_net.state_dict(),
                       os.path.join(checkpoints_dir, 'policynet_checkpoint.pt'))
            torch.save(meta_controller.target_net.state_dict(),
                       os.path.join(checkpoints_dir, 'targetnet_checkpoint.pt'))
            with open(pjoin(checkpoints_dir, 'memory_episode_{0}.pkl'.format(episode)), 'wb') as f:
                dill.dump(meta_controller.memory.memory, f)
            # with open(pjoin(checkpoints_dir, 'memory_weights_episode_{0}.pkl'.format(episode)), 'wb') as f:
            #     dill.dump(meta_controller.memory.weights, f)

            meta_controller_dict = {'gammas': meta_controller.gammas,
                                    'gamma_delay_episode': meta_controller.gamma_delay_episodes,
                                    'gamma_episodes': meta_controller.gamma_episodes,
                                    'all_gammas_ramped_up': meta_controller.all_gammas_ramped_up}
            with open(pjoin(checkpoints_dir, 'meta_controller.pkl'), 'wb') as fp:
                pickle.dump(meta_controller_dict, fp)

            train_dict = {'episode': episode}
            with open(pjoin(checkpoints_dir, 'train.pkl'), 'wb') as fp:
                pickle.dump(train_dict, fp)
            for q_i, Q in enumerate(meta_controller.saved_target_nets):
                torch.save(Q.state_dict(), os.path.join(checkpoints_dir, 'Q_{0}_checkpoint.pt'.format(q_i)))

            # save memory, epsilon, gammas, Qs,
    return meta_controller, res_folder
