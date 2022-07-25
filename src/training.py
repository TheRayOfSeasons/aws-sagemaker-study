import os
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from src.settings import DEFAULT_DOTFILE_PATH, DEFAULT_PICKLE_PATH
from src.settings import MODEL_FILENAME
from src.utils import dump_joblib
from src.utils import dump_pickle


def train(
        training_data,
        dump_name,
        visualize=False,
        hyperparameters={},
        model_dir=DEFAULT_PICKLE_PATH,
        model_filename=MODEL_FILENAME):
    """
    Trains the AI then dumps the model created into a file.
    """

    [X, y] = pickle.load(open(training_data, 'rb'))

    model = DecisionTreeClassifier()
    if hyperparameters:
        model.set_params(**hyperparameters)
    model.fit(X=X.values, y=y)

    # export model as joblib
    dump_joblib(model, filename=model_filename, source_path=model_dir)

    # export model as pickle
    dump_pickle(model, filename=model_filename, source_path=model_dir)

    if visualize:
        # export visualization of decision making by model
        tree.export_graphviz(
            model,
            out_file=os.path.join(DEFAULT_DOTFILE_PATH, f'{dump_name}.dot'),
            feature_names=['age', 'gender'],
            class_names=sorted(y.unique()),
            label='all',
            rounded=True,
            filled=True
        )

    return model
