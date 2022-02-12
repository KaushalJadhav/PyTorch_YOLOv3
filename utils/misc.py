import argparse
import os 
import torch
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg',help='config file.')
    parser.add_argument('--checkpoint',default=None, type=str,help='pytorch checkpoint file path')
    parser.add_argument('--wandb_checkpoint',default=None, type=str,help='path to the checkpoint saved on WandB.')
    return parser.parse_args()

def iscuda(cfg):
    cuda = torch.cuda.is_available() and (cfg['NUM_GPUS']>0)
    if cuda:
        print("Using cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['GPU_IDS'])
    return cuda 

def load_cfg(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    print("successfully loaded config file: ", cfg)
    return cfg 
        