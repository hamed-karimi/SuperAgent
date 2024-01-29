import os.path
from datetime import datetime

from AgentExplorationFunctions import *
from MetaControllerTesting import MetaControllerVisualizer
from MetaControllerTraining import training_meta_controller
from ObjectFactory import ObjectFactory
from Utilities import Utilities
# from TestMetaController import test_meta_controller

utility = Utilities()
params = utility.params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta_controller_dir = params.PRETRAINED_META_CONTROLLER
meta_controller = None
if params.TRAIN_META_CONTROLLER:
    start_time = datetime.now()
    meta_controller, meta_controller_dir = training_meta_controller(utility)
    meta_controller_model_path = os.path.join(meta_controller_dir, 'meta_controller_model.pt')
    torch.save(meta_controller.target_net.state_dict(), meta_controller_model_path)
else:
    object_factory = ObjectFactory(utility)
    utility.make_res_folder(pre_created=params.PRETRAINED_META_CONTROLLER)
    meta_controller = object_factory.get_meta_controller(pre_trained_weights_path=params.PRETRAINED_META_CONTROLLER)

meta_controller_visualizer = MetaControllerVisualizer(utility)
meta_controller_visualizer.get_goal_directed_actions(meta_controller)
# if params.TEST_Q_VALUES:
#     test_meta_controller(utility, meta_controller_dir)
