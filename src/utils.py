import argparse
import os
import pickle

import joblib

from src.settings import DEFAULT_PICKLE_PATH


def dump_pickle(data, filename, source_path=DEFAULT_PICKLE_PATH):
    """
    Dumps a piece of data into a pickle.
    """

    target_path = os.path.join(source_path, f'{filename}.pickle')
    pickle.dump(data, open(target_path, 'wb'))


def dump_joblib(data, filename, source_path=DEFAULT_PICKLE_PATH):
    """
    Dumps a piece of data into a joblib.
    """

    target_path = os.path.join(source_path, f'{filename}.joblib')
    joblib.dump(data, target_path)


def load_pickle(filename, source_path=DEFAULT_PICKLE_PATH):
    """
    Loads a piece of data from a pickle.
    """

    target_path = os.path.join(source_path, f'{filename}.pickle')
    return pickle.load(open(target_path, 'rb'))


def load_joblib(filename, source_path=DEFAULT_PICKLE_PATH):
    """
    Loads a piece of data from a joblib.
    """

    target_path = os.path.join(source_path, f'{filename}.joblib')
    return joblib.load(target_path)


def parse_args():
    """
    Parse arguments.

    Reference: https://github.com/learn-mikegchambers-com/aws-mls-c01/blob/master/8-SageMaker/SageMaker-Script-Mode/script/script.py
    """
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    # We don't use these but I left them in as a useful template for future development
    parser.add_argument('--copy_X',        type=bool, default=True)
    parser.add_argument('--fit_intercept', type=bool, default=True)
    parser.add_argument('--normalize',     type=bool, default=False)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()
