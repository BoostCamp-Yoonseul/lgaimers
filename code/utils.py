import random
import os
import numpy as np
import argparse
import omegaconf
import torch

def config_parser():
   parser = argparse.ArgumentParser()
   parser.add_argument('--config', type = str, default = '../config/config.yaml')
   args = parser.parse_args()

   config = omegaconf.OmegaConf.load(args.config)
   return config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
