README for CSC 580 Project
------------

Author: Aniket Panda
Date: December 8, 2023

## Notes

To run program, simply ensure all included files are within the same directory and type
python3 [script_name.py] as one normally would. A number of dependencies such as
TensorFlow Keras and Scikit-Learn are used. However, the only unusual one is SciKeras,
a Keras wrapper with a Scikit-Learn interface. The source to install it is included in
the References section below. The program was tested using macOS Monterey Version 12.7 
as an OS. When running mma_tuning.py, it is advised that the 10-fold cross validation 
blocks are commented out as these take a very long time. There are no bugs or other 
noteworthy idiosyncracies as far as I know.

## Included files

* README.md -- this file
* data.csv -- holds all data for the project
* mma_original.py -- a replication of the original project that inspired this one.
* mma_features.py -- same as mma_original.py, but with added features.
* mma_tuning.py -- builds on mma_features.py with hyperparameter tuning.

## References

* inspired by https://www.kaggle.com/code/dbsimpson/mma-ml-model-fight-predictions-ufc-259/notebook
* data from https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset?select=ufc-master.csv
* SciKeras from https://adriangb.com/scikeras/stable/index.html

