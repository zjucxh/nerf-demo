from utils import load_data 
from argparser import parse_arguments, read_config_file
import numpy as np 
import torch
import logging

# For repeatability
seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    args = parse_arguments()
    config_data = read_config_file(args)

    images, poses, focal = load_data(config_data=config_data, scene='bulldozer')
    logging.debug(f' images maximum pixel val : {np.max(images.numpy(), axis=None)}')
    print('Done')
