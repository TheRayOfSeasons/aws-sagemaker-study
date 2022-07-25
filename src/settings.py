import os


def auto_create_directory(path):
    """
    Automatically creates a directory if it does not exist.
    """

    try:
        os.mkdir(path)
    except FileExistsError:
        pass


BASE_DIR = os.path.dirname((os.path.dirname(__file__)))
PROJECT_FOLDER = os.path.expanduser(BASE_DIR)

DEFAULT_PICKLE_PATH = os.path.join(BASE_DIR, 'pickles')
auto_create_directory(DEFAULT_PICKLE_PATH)

DATASET_CSV_SOURCE = os.path.join(BASE_DIR, 'data/music.csv')
DATASET_PICKLE_NAME = 'train'
DATASET_DUMP_PATH = os.path.join(
    DEFAULT_PICKLE_PATH,
    f'{DATASET_PICKLE_NAME}.pickle'
)

MODEL_FILENAME = 'model'

DEFAULT_DOTFILE_PATH = os.path.join(BASE_DIR, 'dotfiles')
auto_create_directory(DEFAULT_DOTFILE_PATH)
