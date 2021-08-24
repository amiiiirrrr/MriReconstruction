"""
config.py is written for define config parameters
"""

import json
import os
import argparse

class BaseConfig:
    """
    This class set static paths and other configs.
    Args:
    argparse :
    The keys that users assign such as sentence, tagging_model and other statictics paths.
    Returns:
    The configuration dict specify text, statics paths and controller flags.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.run()

    def run(self):
        """
        The run method is written to define config arguments
        :return: None
        """

        ######################### Paths ###############################
        self.parser.add_argument(
            '--batch_size', type=int, default=8,
        )
        self.parser.add_argument(
            '--early_stopping_num', type=int, default=10,
        )
        self.parser.add_argument(
            '--lr', type=int, default=0.0001,
        )
        self.parser.add_argument(
            '--lr_decay', type=int, default=0.5,
        )
        self.parser.add_argument(
            '--decay_every', type=int, default=10,
        )
        self.parser.add_argument(
            '--beta1', type=int, default=0.9,
            help='beta1 in Adam optimiser'
        )
        self.parser.add_argument(
            '--n_epoch', type=int, default=100,
        )
        self.parser.add_argument(
            '--sample_size', type=int, default=50,
        )
        self.parser.add_argument(
            '--g_alpha', type=int, default=15,
            help='weight for pixel loss'
        )
        self.parser.add_argument(
            '--g_gamma', type=int, default=0.0025,
            help='weight for perceptual loss'
        )
        self.parser.add_argument(
            '--g_beta', type=int, default=0.1,
            help='weight for frequency loss'
        )
        self.parser.add_argument(
            '--g_adv', type=int, default=1,
            help='weight for frequency loss'
        )
        self.parser.add_argument(
            '--seed', type=int, default=100,
        )
        self.parser.add_argument(
            '--epsilon', type=int, default=0.00001,
        )

        ######################### Paths ###############################
        self.parser.add_argument(
            '--data_saving_path', type=str, default='data/MICCAI13_SegChallenge/',
        )
        self.parser.add_argument(
            '--training_data_path', type=str, default="data/MICCAI13_SegChallenge/Training_100",
        )
        self.parser.add_argument(
            '--testing_data_path', type=str, default="data/MICCAI13_SegChallenge/Testing_50",
        )
        self.parser.add_argument(
            '--validation_data_path', type=str, default= "data/MICCAI13_SegChallenge/Validation",
        )
        self.parser.add_argument(
            '--VGG16_path', type=str, default=os.path.join(
            'trained_model', 'VGG16', 'vgg16_weights.zip'),
        )
        self.parser.add_argument(
            '--training_data_path', type=str, default=os.path.join(
            'data', 'MICCAI13_SegChallenge', 'training.pickle'),
        )
        self.parser.add_argument(
            '--val_data_path', type=str, default=os.path.join(
            'data', 'MICCAI13_SegChallenge', 'validation.pickle'),
        )
        self.parser.add_argument(
            '--testing_data_path', type=str, default=os.path.join(
            'data', 'MICCAI13_SegChallenge', 'testing.pickle'),
        )

        ######################### Model Parameters ###############################
        self.parser.add_argument(
            '--model', type=str, default='unet_refine', help='unet, unet_refine'
        )
        self.parser.add_argument(
            '--mask', type=str, default='radial2d', help='gaussian1d, gaussian2d, radial2d'
        )
        self.parser.add_argument(
            '--maskperc', type=int, default='30', help='10,20,30,40,50'
        )
        self.parser.add_argument(
            '--mask_Gaussian1D_path', type=str, default=os.path.join('mask', 'Gaussian1D'),
            help='10,20,30,40,50'
        )
        self.parser.add_argument(
            '--mask_Gaussian2D_path', type=str, default=os.path.join('mask', 'Gaussian2D'),
            help='10,20,30,40,50'
        )
        self.parser.add_argument(
            '--mask_Radial2D_path', type=str, default=os.path.join('mask', 'Radial2D'),
            help='10,20,30,40,50'
        )

    def get_args(self):
        """
        The get_args method is written to return config arguments
        :return: dict
        """
        return self.parser.parse_args()

def log_config(filename, cfg):
    """
    The log_config method is written to save config arguments
    :param filename: str
    :param cfg: dict
    :return:
        None
    """
    with open(filename, 'w') as fff:
        fff.write("================================================\n")
        fff.write(json.dumps(cfg, indent=4))
        fff.write("\n================================================\n")
