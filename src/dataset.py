import pickle

import pandas as pd

from src.settings import DATASET_CSV_SOURCE
from src.settings import DATASET_DUMP_PATH


def dump_dataset():
    """
    Dumps the extracted dataset from
    the CSV into a pickle.
    """

    music_dataframe = pd.read_csv(DATASET_CSV_SOURCE)

    # input set
    # without genre
    X = music_dataframe.drop(columns=['genre'])

    # output set
    # genre only
    y = music_dataframe['genre']

    pickle.dump([X,y], open(DATASET_DUMP_PATH, 'wb'))
