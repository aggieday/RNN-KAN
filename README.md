# A Physics-Guided RNN-KAN for Multi-Step Prediction of Heat Pump Operation State
## Introduction
This repository provides the new modeling method of the operating state of heat pumps based on the RNN-KAN. Besides, three different methods for training the model are also included.  
The KAN used in this project is based on the efficient-kan, and its original code is available [here](https://github.com/Blealtan/efficient-kan?tab=readme-ov-file).
## Contents
The information of this repository file is as follows:
* __kan.py:__
the construction file for the KAN model.
* __RNN_KAN_layer.py:__
the construction file for the RNN-KAN model.
* __RNN_KAN.py:__
the file that provides functions for training, predicting, and evaluating the RNN-KAN model.
* __training:__
the folder that provides the complete prediction workflow for three different model training methods.
    * batch_training.py:the complete workflow under batch training.
    * online_training.py: the complete workflow under online training.
    * error_based_training.py: the complete workflow under prediction error based training.
* __example:__
the folder that provides example data and the complete process of using batch training for prediction.
    * batch_training.ipynb: the complete prediction code.
    * test_data.csv: test set data
    * training_data.npy: training set data
    * prediction_results.csv: prediction results

## Notice
The related work can be found in the Paper [*"A Physics-Guided RNN-KAN for Multi-Step Prediction of Heat Pump Operation State"*](), published in *Energy* - The International Journal by Elsevier.  
If there is further citation of this work, please cite this paper kindly.