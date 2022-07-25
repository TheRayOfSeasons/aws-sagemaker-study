from src.dataset import dump_dataset
from src.enums import Gender
from src.settings import DATASET_DUMP_PATH
from src.settings import MODEL_FILENAME
from src.training import train
from src.utils import load_pickle


TEST_DATA = [
    [21, Gender.MALE],
    [22, Gender.FEMALE],
    [32, Gender.FEMALE],
]


def test():
    """
    Tests the training process and the model itself.
    """

    # arrange
    dump_dataset()
    train(
        training_data=DATASET_DUMP_PATH,
        dump_name=MODEL_FILENAME,
        visualize=True
    )
    model = load_pickle(MODEL_FILENAME)

    # act
    predictions = model.predict(TEST_DATA)

    # assert
    expected_values = ['HipHop', 'Dance', 'Classical']
    assert len(TEST_DATA) == len(expected_values)
    for i in range(len(expected_values)):
        prediction = predictions[i]
        expected = expected_values[i]
        result = prediction == expected
        print(
            f'Prediction: {prediction} | '
            f'Expected: {expected} | '
            f'Result: {result}'
        )
        assert prediction == expected
