# AWS Sagemaker study

This repository contains a study project for using AWS Sagemaker. This can also be used as a reference or as a boilerplate.

# Setup (In Sagemaker Studio)

1. Launch Sagemaker studio from the AWS console.
2. Open the git tab on the menu bar, then click clone repository.
   1. For more information, see this (video)[https://www.youtube.com/watch?v=FqU13I_E0jk].
3. Open a terminal in Sagemaker studio.
4. `cd` into the repository.
5. Create a python virtualenv with the python version 3.7.0.
6. Activate the virtualenv.
7. Run `pip install -r requirements.txt`
8. Run `python manage.py deploy`.

# Optional setup steps

These setup steps involve the full deployment of the codebase.

1. Open a terminal in your local device (not in Sagemaker Studio!).
2. Clone and `cd` into the repositroy.
3.  `cd` into the `lambda` directory and run `sls deploy`.
    1.  If `sls` is not installed, install `serverless` globally via npm. `npm install -g serverless`.

After deploying the lambda, you may now invoke predictions from the model through the API Gateway endpoint.

# File breakdown

```bash
├── README.md
├── __init__.py
├── data -> "Raw dataset"
│   └── music.csv
├── lambda -> "The Lambda that exposes the Sagemaker endpoint as an HTTP API"
│   ├── handler.py
│   └── serverless.yml
├── manage.py -> "The main script that acts as a CLI."
├── dotfiles -> "A directory containing dotfiles for visualization"
├── pickles -> "A directory containing pickles and joblibs"
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── dataset.py -> "Functionalities for extracting data from the source file"
│   ├── endpoint.py -> "Functionalities for setting up the Sagemaker endpoint"
│   ├── enums.py
│   ├── settings.py -> "Constants and configs"
│   ├── test.py -> "Test script"
│   ├── training.py -> "In charge of training the AI"
│   └── utils.py
└── train.py -> "The entrypoint used by training jobs and Sagemaker endpoints."
```

# Commands

1. `python manage.py test` - Trains the model then tests it for predictions.
2. `python manage.py deploy` - Trains the model via training jobs, then deploys it in a Sagemaker endpoint. This only works in Sagemaker studio, and not in local.
