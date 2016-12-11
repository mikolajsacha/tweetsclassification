Twitter classificator
==============================

Implementation of chosen Machine Learning algorithms for assigning tweets to categories.

Project Organization
------------

    ├── LICENSE
    ├── Makefile             <- Makefile with commands like `make data` or `make train`
    ├── README.md            <- The top-level README for developers using this project.
    ├── data
    │   └── dataset_n_name   <- Folder for one of the datasets
    │       ├── external     <- Data from third party sources.
    │       └── processed    <- The final, canonical data sets for modeling.
    │
    ├── models               <- Trained and serialized models
    │   ├── features         <- Trained features sets (in human readable form)
    │   └── word_embeddings  <- Trained word embeddings (in binary form)
    │
    ├── notebooks            <- Jupyter notebooks. Not yet created.
    │
    ├── references           <- Data dictionaries, manuals, and all other explanatory materials. Not yet created
    │
    ├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt     <- The requirements file for reproducing the analysis environment
    │
    ├── src                  <- Source code for use in this project.
    │    │
    │    ├── data            <- Scripts to generate preprocessed data
    │    │   └── make_dataset.py
    │    │
    │    ├── features        <- Scripts to turn raw data into features for modeling
    │    │   ├── word_embeddings     <- Word embeddings
    │    │   ├── sentence_embeddings <- Scripts for embedding word sets into set vectors
    │    │   └── build_features.py   <- Script for building features for a given word end sentence embedding
    │    │
    │    ├── models          <- Scripts to train and test models
    │    │   ├── algorithms          <- Wrappers around a few of data classification algorithms
    │    │   └── model_testing       <- Scripts for testing model accuracy
    │    │
    │    └── visualization   <- Scripts to create exploratory and results oriented visualizations
    │        └── interactive.py      <- Script for testing models by predicting sentences interactively
    │    
    └── summaries            <- Results of tests of models' performance in textual form
    
