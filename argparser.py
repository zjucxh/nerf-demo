import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process nerf configuration parameters')
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    return args


def read_config_file(config_file):
    with open(config_file.config, 'r') as fd:
        config_data = yaml.safe_load(fd)
    return config_data

