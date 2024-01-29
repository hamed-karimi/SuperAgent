import json
from types import SimpleNamespace
import os
import shutil
import pickle
from datetime import datetime
import numpy as np


class Utilities:
    def __init__(self):
        self.res_folder = None
        with open('./Parameters.json', 'r') as json_file:
            self.params = json.load(json_file,
                                    object_hook=lambda d: SimpleNamespace(**d))

    def make_res_folder(self, sub_folder='', pre_created=''):
        if pre_created != '':
            folder = None
            dirname = pre_created
        else:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder = 'tr{0}'.format(now)
            dirname = os.path.join(folder, sub_folder)

        if folder is not None and os.path.exists(folder) and not os.path.exists(dirname):
            os.mkdir(dirname)
        elif not os.path.exists(dirname):
            os.makedirs(dirname)
        self.res_folder = dirname
        shutil.copy('./Parameters.json', self.res_folder)
        return dirname

    def get_last_episode(self):
        if self.params.CHECKPOINTS_DIR != "":
            with open(os.path.join(self.params.CHECKPOINTS_DIR, 'train.pkl'), 'rb') as f:
                train_dict = pickle.load(f)
                return train_dict['episode']
        return -1
