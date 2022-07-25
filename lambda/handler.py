import json
import os

import boto3

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    """
    POST.

    A handler that acts as the controller
    for delegating the data to the AI from
    Sagemaker, then returns it as an API
    response.

    Acceptable JSON body:

    {
        "data": [
            [21, 1],
            [22, 0],
            [32, 0]
        ]
    }

    Structure:
    [{age}, {gender - 1 is male, 0 is female}]
    """

    body = json.loads(event['body'])
    payload = body['data']

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    result = json.loads(response['Body'].read().decode())

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
