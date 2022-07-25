"""
Must be run from Sagemaker Studio or a Sagemaker Notebook.
"""

import os

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

from src.dataset import dump_dataset
from src.settings import DATASET_DUMP_PATH
from src.settings import DATASET_PICKLE_NAME


BUCKET_NAME = 'sagemaker-learning-bucket-5678'


def setup(auto_delete=False):
    """
    Delegates the training of a model to a training
    job, then deploys it to a Sagemaker endpoint.

    Reference: https://github.com/learn-mikegchambers-com/aws-mls-c01/blob/master/8-SageMaker/SageMaker-Script-Mode/SageMaker-Script-Mode.ipynb
    """

    dump_dataset()

    S3 = boto3.Session().resource('s3')
    bucket = S3.Bucket(BUCKET_NAME)

    s3_prefix = 'script-mode-workflow'
    pickle_s3_prefix = f'{s3_prefix}/pickle'
    pickle_s3_uri = f's3://{BUCKET_NAME}/{s3_prefix}/pickle'
    pickle_train_s3_uri = f'{pickle_s3_uri}/train'

    # Upload dataset.pickle to the bucket. This is so
    # the estimator can access the dataset when it
    # runs on an EC2 instance encapsulated by Sagemaker.
    bucket_path = os.path.join(
        pickle_s3_prefix,
        f'{DATASET_PICKLE_NAME}.pickle'
    )
    bucket.Object(bucket_path).upload_file(DATASET_DUMP_PATH)

    role = sagemaker.get_execution_role()

    train_instance_type = 'ml.m5.large'

    estimator_parameters = {
        'entry_point': 'train.py',
        'source_dir': '.',
        'framework_version': '0.23-1',
        'py_version': 'py3',
        'instance_type': train_instance_type,
        'instance_count': 1,
        'hyperparameters': {}, # use this to pass hyperparameters to the training model.
        'role': role,
        'base_job_name': 'music-model',
    }

    estimator = SKLearn(**estimator_parameters)

    estimator.fit({
        'train': pickle_train_s3_uri
    })

    sklearn_predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name='music-recommender-endpoint'
    )

    if auto_delete:
        # Use this to undeploy the endpoint:
        sklearn_predictor.delete_endpoint(True)
