import model.model as model
import numpy as np
import os
import torch

def load(model_name, model_file=None):
    lgbm_path = '/home/steven/loss-landscape/flight/model/lightgbm.npz'
    if model_name == 'flight_random':
        net = model.Model(np.load(lgbm_path), 4, True, True, 0.1, None)
        #tree_params, opt_level, random_init=False, train_ohe=False, do, do_ohe
        print(model_name)
    elif model_name == 'flight_finetune':
        net = model.Model(np.load(lgbm_path), 4, False, True, 0.1, None)
        print(model_name)
    
    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        net.load_state_dict(stored['model_state_dict'])
        """
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)
        """
    net.float()
    return net