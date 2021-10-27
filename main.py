import argparse
import os
import json
import random

import numpy as np
import torch
import yaml

from train import Trainer

class Config:
    def __init__(self, raw):
        for k,v in raw.items():
            setattr(self,k,v)

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--exp_name", default='debug')
    cli.add_argument("--seed", default=19)
    cli.add_argument("--device", default='cpu')
    opts = cli.parse_args()
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    np.random.seed(opts.seed)
    config = Config(yaml.safe_load(open(f"config.yml", "r")))

    t = Trainer(config,opts.device,opts.exp_name)
    t.train()



if __name__ == '__main__':
    main()
