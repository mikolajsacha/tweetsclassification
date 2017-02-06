Twitter classificator
==============================

Implementation of chosen Machine Learning algorithms for assigning tweets to categories.
You can read description of the project in Jupyter notebook in root project directory.

Project Organization
------------

    ├── Notebook.ipynb       <- Jupyter notebook. You probably want to read this.
    ├── LICENSE
    ├── Makefile             <- Makefile with commands like `make data` or `make train`
    ├── README.md            <- The top-level README for developers using this project.
    ├── data
    │   └── dataset_n_name   <- Folder for one of the datasets
    │       ├─ external      <- Data from third party sources.
    │       ├─ processed     <- The final, canonical data sets for modeling.
    │       │
    │       gathered_dataset <- Folder for the dataset that I manually gathered using Twitter API
    │       ├─ external   
    │       ├─ processed
    │       └─ tweets_mining <- Temporary files with data gathered using Twitter API
    │
    ├── models               <- Trained and serialized models
    │   └── word_embeddings  <- Pre-trained word embeddings
    │       ├─ glove_twitter <- Default location for 'glove.twitter27B.200d.txt' embedding file (see notebook)
    │       └─ google        <- Default location for 'GoogleNews-vectors-negative300.bin' embedding file (see notebook)
    │
    ├── reports              <- Generated analysis
    │   └── figures          <- Generated visualizations (using matplotlib)
    │
    ├── requirements.txt     <- The requirements file for reproducing the analysis environment
    │
    ├── src                  <- Source code for use in this project.
    │    │
    │    ├── data            <- Scripts to generate preprocessed data
    │    │   ├── data_gathering      <- Scipts for mining Twitter API
    │    │   └── dataset.py          <- Basic methods for handling the data set
    │    │
    │    ├── features        <- Scripts to turn raw data into features for modeling
    │    │   ├── word_embeddings     <- Scipts for loading Word Embededings
    │    │   ├── sentence_embeddings <- Scripts for embedding word embeddings into set vectors
    │    │   └── build_features.py   <- Script for building features for a given word embedding end sentence embedding
    │    │
    │    ├── models          <- Scripts to train and test models
    │    │   ├── algorithms          <- Wrappers around a few of data classification algorithms
    │    │   └── model_testing       <- Scripts for testing model accuracy using grid-search
    │    │
    │    └── visualization   <- Scripts to create exploratory and results oriented visualizations
    │        └── interactive.py      <- Script for testing models by predicting sentences interactively
    │    
    └── summaries            <- Results of tests of models' performance in textual form
    
