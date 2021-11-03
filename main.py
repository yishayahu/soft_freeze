import argparse

import random

import numpy as np
import torch
import yaml

from train import Trainer

class Config:
    def __init__(self, raw):
        for k,v in raw.items():
            setattr(self,k,v)

def seed_all(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--exp_name", default='debug')
    cli.add_argument("--seed", default=19)
    cli.add_argument("--device", default='cpu')
    opts = cli.parse_args()
    seed_all(opts.seed)
    config = Config(yaml.safe_load(open(f"config.yml", "r")))

    t = Trainer(config,opts.device,opts.exp_name)
    t.train()



if __name__ == '__main__':
    main()
