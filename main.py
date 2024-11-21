from argparser import parse_arguments, read_config_file
from utils import load_data, get_rays

if __name__=='__main__':
    args = parse_arguments()
    config_data = read_config_file(args)

    images, poses, focal_length = load_data(config_data)


    print('Done')