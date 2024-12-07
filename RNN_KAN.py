import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from RNN_KAN_layer import RNN_KANModel
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

def train(trainset_input, trainset_output, hidden_size=30, kan_hidden_size=30, grid_size=5, learning_rate=0.001, weight_decay=1e-4, num_epochs=150, batch_size=32):
    # normalize data
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    trainset_input = input_scaler.fit_transform(trainset_input.reshape(-1, trainset_input.shape[2])).reshape(trainset_input.shape)
    trainset_output = output_scaler.fit_transform(trainset_output)

    input_tensor = torch.tensor(trainset_input, dtype=torch.float32)
    output_tensor = torch.tensor(trainset_output, dtype=torch.float32)
    dataset = TensorDataset(input_tensor, output_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    input_size = trainset_input.shape[2]
    output_size = trainset_output.shape[1]

    model = RNN_KANModel(input_size, hidden_size, kan_hidden_size, output_size, grid_size)

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # trian the model
    best_train_loss = 10.0
    best_test_loss = 10.0
    his_test_loss = 10.0
    for epoch in range(num_epochs):
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
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()

        # calculate average loss
        test_loss /= len(test_loader)
        if test_loss > his_test_loss*1.1:
            break
        his_test_loss = min(test_loss, his_test_loss)
        best_test_loss = min(best_test_loss, test_loss)
        
        # scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    print(f'Best Train Loss: {best_train_loss:.4f}, Best Test Loss: {best_test_loss:.4f}')
    return model, input_scaler, output_scaler

def predict(model, data, input_scaler, output_scaler):
    # data=[squence*input_size]
    input = input_scaler.transform(data).reshape(1, data.shape[0],-1)
    input_tensor = torch.tensor(input, dtype=torch.float32)
    model.eval()
    with torch.no_grad(): 
        pred = model(input_tensor)
        pred_inv = output_scaler.inverse_transform(pred.numpy())
    return pred, pred_inv

def evaluate(train_data_set, df, K, input_loc, output_loc, hidden_size=30, grid_size=5):
    trainset_input = train_data_set[:, :, input_loc].astype(float)
    trainset_output = train_data_set[:, -1, output_loc].astype(float)
    model, input_scaler, output_scaler = train(trainset_input, trainset_output, hidden_size, grid_size)
    pred_list = []
    for i in range(len(df)-K):
        data = df.iloc[i:i+K+1,input_loc].values
        _, pred_inv = predict(model, data, input_scaler, output_scaler)
        pred_list.append(pred_inv[0])
    pred_list = np.array(pred_list).reshape(-1,len(output_loc))
    true_list = df.iloc[K:,output_loc].values
    return pred_list, true_list, model