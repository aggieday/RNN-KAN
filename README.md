This repository provides the RNN-KAN-based model code for predicting the operating state of heat pumps, and in provides three different methods for training the model. 
Among them, the KAN network is efficientkan, and its original code is available by https://github.com/Blealtan/efficient-kan?tab=readme-ov-file.
The information of this repository file is as follows:
  kan.py:  construction file for the KAN model.
  RNN_KAN_layer.py: construction file for the RNN-KAN model.
  RNN_KAN.py: provides functions for training, predicting, and evaluating the RNN-KAN model.
  training: provides the complete prediction code for three different model training methods.
    batch_training: batch training followed by prediction.
    online_training: prediction after update training.
    error_based_training: Prediction after update training based on prediction error.
example: provides example data and the complete process of using batch training for prediction.
    batch_training.ipynb: the complete prediction code.
    test_data.csv: test set data
    training_data.npy: training set data
    prediction_results.csv: prediction results

Related work can be found at .
If there is further citation of this work, please cite it in the following format:
