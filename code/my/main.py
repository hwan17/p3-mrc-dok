import time
import wandb
import argparse
from prettyprinter import cpprint

from utils import *
from engine import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='/opt/ml/my_code/config_defaults.yml')
    parser.add_argument('--config', type=str, default="trial1")
    parser.add_argument('--mode', type=str, default='train', help = 'train, inference, final_train')
    args = parser.parse_args()
    
    cfg = YamlConfigManager(args.config_file_path, args.config)
    
    name = '(' + cfg.values.mrc.model_name + ')' + ' ' + get_timestamp()
    
    if args.mode == 'train' and cfg.values.train.mrc == 'mrc':
      run = wandb.init(project='MRC', entity='jlee621', name = name, reinit = False)
      wandb.config.update(cfg) # add all the arguments as config variables

    cpprint(cfg.values, sort_dict_keys = False)
    print('\n')
    engine(cfg, args)
    
    if args.mode == 'train' and cfg.values.train.mrc == 'mrc':
      run.finish()