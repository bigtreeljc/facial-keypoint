import torch
import torch.nn as nn

import yaml
from easydict import EasyDict as edict
import logging


class conf:
    def __init__(self, yaml_file: str):
        logging.debug("yaml_file {} type {}".format(
            yaml_file, type(yaml_file)))
        with open(yaml_file, 'r') as f:
            conf_dic = edict(yaml.load(f))
        for k, v in conf_dic.items():
            setattr(self, k, v)

class keypoint_regression_model(nn.Module):
    def __init__(self, width=96, height=96, num_keypoints=30, 
            n_channel=1):
        '''
            form a network that will predict 30 keypoint coord
        '''
        super(keypoint_regression_model, self).__init__()
        self.conv_1 = nn.Sequential(
                nn.Conv2d(n_channel, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )

        self.conv_2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )

        self.conv_3 = nn.Sequential(
                nn.Conv2d(64, 96, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        '''
            batch_size * 96 * 12 * 12
        '''
        self.fcs = nn.Sequential(
                nn.Linear(96*12*12, 1000),
                nn.Linear(1000, num_keypoints)
                )

    def forward(self, x):
        # x = self.conv_3(self.conv_2(self.conv_1(x)))
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 96*12*12)
        y = self.fcs(x)
        return y.clamp(min=-1.0, max=1.0)
