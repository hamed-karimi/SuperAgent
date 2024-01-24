from Agent import Agent
from Environment import Environment
from Controller import Controller
from MetaController import MetaController
import inspect
from copy import deepcopy


class ObjectFactory:
    def __init__(self, utility):
        self.agent = None
        self.environment = None
        self.controller = None
        self.meta_controller = None
        self.params = utility.params

    def get_agent(self, pre_location, preassigned_mental_states, preassigned_mental_states_change):
        agent = Agent(self.params.HEIGHT, self.params.WIDTH, n=self.params.OBJECT_TYPE_NUM,
                      prob_init_needs_equal=self.params.PROB_OF_INIT_NEEDS_EQUAL, predefined_location=pre_location,
                      preassigned_mental_states=preassigned_mental_states,
                      preassigned_state_change=preassigned_mental_states_change,
                      lambda_satisfaction=self.params.LAMBDA_SATISFACTION,
                      epsilon_function=self.params.EPSILON_FUNCTION)
        # self.agent = deepcopy(agent)
        return agent

    def get_environment(self, agent, few_many,
                        pre_assigned_object_reward,
                        pre_located_objects_num,
                        pre_located_objects_location,
                        prohibited_object_location):  # pre_located_objects is a 2D list
        curr_frame = inspect.currentframe()
        call_frame = inspect.getouterframes(curr_frame, 2)
        # if 'meta_controller' in call_frame[1][3]:
        #     num_objects = self.params.OBJECT_TYPE_NUM
        # else:
        #     num_objects = 1
        env = Environment(few_many, self.params.HEIGHT, self.params.WIDTH, agent,
                          num_object=self.params.OBJECT_TYPE_NUM,
                          pre_assigned_object_reward=pre_assigned_object_reward,
                          pre_located_objects_num=pre_located_objects_num,
                          pre_located_objects_location=pre_located_objects_location,
                          prohibited_object_location=prohibited_object_location)
        # self.environment = deepcopy(env)
        return env

    def get_controller(self):
        controller = Controller(self.params.HEIGHT,
                                self.params.WIDTH)

        # self.controller = deepcopy(controller)
        return controller

    def get_meta_controller(self, pre_trained_weights_path=''):
        meta_controller = MetaController(self.params,
                                         pre_trained_weights_path=pre_trained_weights_path)
        # self.meta_controller = deepcopy(meta_controller)
        return meta_controller

    def get_saved_objects(self):
        return deepcopy(self.agent), deepcopy(self.environment), \
               deepcopy(self.controller), deepcopy(self.meta_controller)
