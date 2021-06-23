Garbage Classification
==============================

The final project for DTU's Special Course: Machine Learning Operations
The objective of our project was to implement a classifier capable of distinquishing the different types of garbage.
Our dataset was downloaded from [kaggle](https://www.kaggle.com/asdasdasasdas/garbage-classification). 
The initial dataset was consist of 6 types of garbage. More precisely: Cardboad, Glass, Metal, Paper, Plastic and Trash.

We also took advantage of the [Kornia](https://github.com/kornia/kornia) framework and performed data augmentantion to the above mentioned dataset.
Overall, 5304 new images were generated and used for training purposes.


Usage
==============================
1. Install all the required packages: `pip install -r requirement.txt`
2. In order to download the dataset: `python src/data/make_dataset.py`
3. Once you have the dataset, initialize the training: `python src/models/train.py`
4. After training, evaluate the model: `python src/models/evaluate.py`
5. If you only want to classify a single image, first, place your desired image in the following folder: `data/external` and then simply `python src/models/predict_model.py [--load_model_from] <image_name>`


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── wandb          <- Generated wandb logs
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── tests
    │   ├── __init__.py    <- Makes src a Python module
    │   └── test_data.py               
    │   └── test_model.py
    │   └── test_training.py
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │   └── distributed_data_loading.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train.py
    │   │   └── evaluate.py
    │   │   └── model.py
    │   │   └── train_optuna.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
