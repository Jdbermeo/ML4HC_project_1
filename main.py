import yaml
import logging
import argparse

from model import train, predict

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,  choices=['train', 'predict'],
                        help='mode to run the script, either train, predict, or evaluate')
    parser.add_argument('--config_file_path', type=str, default='config.yml',
                        help='mode to run the script, either train, predict, or evaluate')

    args_ = parser.parse_args()

    return args_


if __name__ == '__main__':
    # Parse script arguments
    args = parse_arguments()

    # Parse yaml config file parameters

    with open(args.config_file_path) as yaml_file:
        config_params = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if args.mode == 'train':
        logging.basicConfig(filename='training.log', level=logging.DEBUG, filemode='w')
        train.train(data_path_source_dir_=config_params['data_params']['data_dir_path'],
                    training_params=config_params['training_params'],
                    model_params=config_params['model_params'])

    elif args.mode == 'predict':
        logging.basicConfig(filename='prediction.log', level=logging.DEBUG, filemode='w')
        predict.predict(data_path_source_dir_=config_params['data_params']['data_dir_path'],
                        training_params=config_params['training_params'],
                        model_params=config_params['model_params'])

    else:
        raise Exception('Script can only be ran with --mode `train` or `predict`')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
