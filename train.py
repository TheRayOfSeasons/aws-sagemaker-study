"""
The main entry point for training jobs
and Sagemaker endpoints.
"""

import pickle

from src.settings import DATASET_DUMP_PATH
from src.settings import MODEL_FILENAME
from src.training import train
from src.utils import parse_args


def model_fn(model_dir):
    """
    Load the model for inference
    """

    loaded_model = pickle.load(
        open(f'{model_dir}/{MODEL_FILENAME}.pickle', 'rb')
    )
    return loaded_model


def predict_fn(input_data, model):
    """
    Apply model to the incoming request.
    """

    return model.predict(input_data)


if __name__ == '__main__':
    args, _ = parse_args()

    train(
        training_data=DATASET_DUMP_PATH,
        dump_name=MODEL_FILENAME,
        model_dir=args.model_dir
    )
