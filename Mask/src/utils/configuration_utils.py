import argparse
import os
import pprint
import yaml
import logging
import random
import string

from shutil import copyfile


def parse_arguments():
    arguments_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arguments_parser.add_argument('--config', help='yml config path', type=str, required=True)
    args = arguments_parser.parse_args()
    return args.config


def initial_configuration(prevent_logging=False, cfg_path=None):
    if cfg_path is None:
        cfg_path = parse_arguments()

    with open(cfg_path, 'r') as yaml_file:
        params = yaml.load(yaml_file, Loader=yaml.FullLoader)

    experiment_name = "{name}-{challenge}-{id}".format(name=params['experiment']['name'],
                                                       challenge=params['experiment']['challenge'],
                                                       id=id_generator())
    if not prevent_logging:
        create_logger(params['logging']['log_dir'],
                      experiment_name + '.log',
                      console_level=logging.CRITICAL,
                      file_level=logging.NOTSET)

        pp = pprint.PrettyPrinter(indent=4)
        logging.info("Configuration is:\n%s" % pp.pformat(params))
        experiment_path = os.path.join(params['logging']['log_dir'], experiment_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        copyfile(cfg_path, os.path.join(experiment_path, 'config.yml'))
    else:
        print("TEST RUN | Prevent logging and model saving")

    return params, experiment_name


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def create_logger(output_folder, log_name, console_level=logging.ERROR, file_level=logging.WARNING):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_filename = os.path.join(output_folder, log_name)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    file_handler = logging.FileHandler(log_filename, 'a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)

    root_logger = logging.getLogger()
    for hdlr in root_logger.handlers[:]:  # remove all old handlers
        root_logger.removeHandler(hdlr)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)
