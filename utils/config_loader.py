import configparser
import os.path
import platform
# from icecream import ic


def load_config(config_file):
    config = configparser.ConfigParser()
    current_platform = platform.system() # Windows/Linux
    config.read(config_file)
    return config[current_platform]

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
config_file = os.path.join(parent_dir, 'config', 'config.ini')

config = load_config(config_file)

