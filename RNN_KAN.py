import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from RNN_KAN_layer import RNN_KANModel
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

'''
    Train the RNN-KAN model:
    [Input]
    trainset_input: input data, 3Darray with shape=[num_samples, sequence_length, feature_size]
    trainset_output: output data, 3Darray with shape=[num_samples, 1, output_size]
    hidden_size: hidden size of RNN
    kan_hidden_size: hidden size of KAN
    grid_size: grid size of KAN
    learning_rate: learning rate of optimizer
    num_epochs: number of epochs
    batch_size: batch size
    [Output]
    model: trained RNN-KAN model
    input_scaler: input scaler (standard scaler)
    output_scaler: output scaler (standard scaler)
''' 
def train(trainset_input, trainset_output, hidden_size=30, kan_hidden_size=30, grid_size=5, learning_rate=0.001, num_epochs=150, batch_size=32):
    # standardize the data
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    trainset_input = input_scaler.fit_transform(trainset_input.reshape(-1, trainset_input.shape[2])).reshape(trainset_input.shape)
    trainset_output = output_scaler.fit_transform(trainset_output)

    input_tensor = torch.tensor(trainset_input, dtype=torch.float32)
    output_tensor = torch.tensor(trainset_output, dtype=torch.float32)
    dataset = TensorDataset(input_tensor, output_tensor)

    # split the dataset into training set and validation set
    train_size = int(0.8 * len(dataset))
    validate_size = len(dataset) - train_size
    train_dataset, validate_dataset = random_split(dataset, [train_size, validate_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size, shuffle=False)

    input_size = trainset_input.shape[2]
    output_size = trainset_output.shape[1]

    model = RNN_KANModel(input_size, hidden_size, kan_hidden_size, output_size, grid_size)

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_train_loss = 10.0
    best_validate_loss = 10.0
    his_validate_loss = 10.0
    for epoch in range(num_epochs):
        # trian the model
        model.train()
        train_loss = 0.0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        best_train_loss = min(best_train_loss, train_loss)
        
        # validate the model
        model.eval()
        validate_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in validate_loader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                validate_loss += loss.item()

        # calculate average loss
        validate_loss /= len(validate_loader)

        # early stopping
        if validate_loss > his_validate_loss*1.1:
            break

        his_validate_loss = min(validate_loss, his_validate_loss)
        best_validate_loss = min(best_validate_loss, validate_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, validate Loss: {validate_loss:.4f}')
    print(f'Best Train Loss: {best_train_loss:.4f}, Best validate Loss: {best_validate_loss:.4f}')
    return model, input_scaler, output_scaler

'''
    Predict results by the trained RNN-KAN model:
    [Input]
    model: trained RNN-KAN model
    data: input data, 2Darray with shape=[sequence_length*feature_size]
    input_scaler: input scaler (standard scaler)
    output_scaler: output scaler (standard scaler)
    [Output]
    pred: predicted output data, 2Darray with shape=[1, output_size]
    pred_inv: inverse transformed output data, prediction = pred_inv[0]
''' 
def predict(model, data, input_scaler, output_scaler):
    input = input_scaler.transform(data).reshape(1, data.shape[0],-1)
    input_tensor = torch.tensor(input, dtype=torch.float32)
    model.eval()
    with torch.no_grad(): 
        pred = model(input_tensor)
        pred_inv = output_scaler.inverse_transform(pred.numpy())
    return pred, pred_inv

'''
    Evaluate the RNN-KAN model: 
    A function that integrates training and prediction steps to test network performance
    [Input]
    train_data_set: training data set, 3Darray with shape=[num_samples, sequence_length, feature_size+output_size]
    df: test data, DataFrame
    K: sequence length
    input_loc: input location
    output_loc: output location
    hidden_size: hidden size of RNN
    grid_size: grid size of KAN
    [Output]
    pred_list: predicted output data, 2Darray with shape=[num_samples-K, output_size]
    true_list: true output data, 2Darray with shape=[num_samples-K, output_size]
''' 
def evaluate(train_data_set, df, K, input_loc, output_loc, hidden_size=30, grid_size=5):
    # split the train data into input and output
    trainset_input = train_data_set[:, :, input_loc].astype(float)
    trainset_output = train_data_set[:, -1, output_loc].astype(float)

    # train the model
    model, input_scaler, output_scaler = train(trainset_input, trainset_output, hidden_size, grid_size)
    
    # predict the output
    pred_list = []
    for i in range(len(df)-K):
        data = df.iloc[i:i+K+1,input_loc].values
        _, pred_inv = predict(model, data, input_scaler, output_scaler)
        pred_list.append(pred_inv[0])
    
    # return the results
    pred_list = np.array(pred_list).reshape(-1,len(output_loc))
    true_list = df.iloc[K:,output_loc].values
    return pred_list, true_list, model