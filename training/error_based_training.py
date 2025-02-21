import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import RNN_KAN
import time
import warnings
warnings.filterwarnings("ignore")
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def indicator(list, pred_list):
    if list.ndim == 1:
        MAPE = np.mean(np.abs((pred_list-list)/list))
        RMSE = np.sqrt(np.mean((pred_list-list)**2))
    else:
        MAPE, RMSE = [], []
        for i in range(pred_list.shape[1]):
            MAPE.append(np.mean(abs((pred_list[:,i]-list[:,i])/list[:,i])))
            RMSE.append(np.sqrt(np.mean((pred_list[:,i]-list[:,i])**2)))
    print(f'MAPE={MAPE},RMSE={RMSE}')
    return MAPE, RMSE

indi_list, time_list = [], []

# load test data
df = pd.read_csv(data_dir+'data_filename.csv')
df['TIME'] = pd.to_datetime(df['TIME'])
data_for_test = df.set_index('TIME')

# set the update tolerance
rmse_p_cnd_b = 181000
rmse_p_evp_b = 36000
rmse_h_cmp_b = 12200
rmse_h_cnd_b = 27900
rmse_h_evp_b = 134700
tolerance = 0.95

cnd_K = 4
evp_K = 6
cmp_K = 7
K = max(cnd_K, evp_K, cmp_K)

# load training data
train_data_set = np.load(f'{data_dir}/training_data.npy', allow_pickle=True)
train_data_set, _  = train_test_split(train_data_set, test_size=0.8, random_state=42)

hidden_size = (30,30)
kan_hidden_size = 20
grid_size = 10

# prepare the training data (initialization)
cmp_trainset_input = train_data_set[:, :, ['cmp_input_columns']].astype(float)
cmp_trainset_output = train_data_set[:, -1, ['cmp_output_columns']].astype(float)
cnd_trainset_input = train_data_set[:, :, ['cnd_input_columns']].astype(float)
cnd_trainset_output = train_data_set[:, -1, ['cnd_output_columns']].astype(float)
evp_trainset_input = train_data_set[:, :, ['evp_input_columns']].astype(float)
evp_trainset_output = train_data_set[:, -1, ['evp_output_columns']].astype(float)

N = data_for_test.shape[0]
PH =1440
P_cnd, P_evp, H_ref_cnd_out, H_ref_evp_out, H_ref_com_out = [], [], [], [], []
i = 0
start_time = time.time()

# train the model by historical data
cmp_model, cmp_input_scaler, cmp_output_scaler = RNN_KAN.train(cmp_trainset_input, cmp_trainset_output, hidden_size[1], kan_hidden_size, grid_size)
cnd_model, cnd_input_scaler, cnd_output_scaler = RNN_KAN.train(cnd_trainset_input, cnd_trainset_output, hidden_size[0], kan_hidden_size, grid_size)
evp_model, evp_input_scaler, evp_output_scaler = RNN_KAN.train(evp_trainset_input, evp_trainset_output, hidden_size[0], kan_hidden_size, grid_size)
# complete first training

while(i+PH<=N-K):
    training_time = time.time()
    time_list.append(f'{i//PH}:Training time: {training_time-start_time}')
    
    # assign the historical values
    h_cmp_out_array = data_for_test.iloc[0:K-1,'h_cmp_out_column'].values
    h_cnd_out_array = data_for_test.iloc[0:K-1,'h_cnd_out_column'].values
    h_evp_out_array = data_for_test.iloc[0:K-1,'h_evp_out_column'].values
    p_cnd_array = data_for_test.iloc[0:K-1,'p_cnd_column'].values
    p_evp_array = data_for_test.iloc[0:K-1,'p_evp_column'].values

    # for condenser as starting subsystem    
    h_ref_com_out = data_for_test.iloc[K-1,'h_cmp_out_column']
    
    # predict the outputs during the prediction horizon
    for t in range(PH):
        H_ref_com_out.append(h_ref_com_out)
        op_paras = data_for_test.iloc[i+t:i+t+K,'operation_param_colums']
        
        # condenser subsystem
        # predict condenser pressure and the outlet enthalpy
        cnd_rnn_x = op_paras[['Tin','Tout','frequency','EEV']].values[-cnd_K:,:]
        h_cmp_out_array = np.append(h_cmp_out_array, h_ref_com_out)
        cnd_rnn_x = np.hstack((cnd_rnn_x, h_cmp_out_array[-cnd_K:].reshape(cnd_K,1)))
        cnd_y, cnd_y_inv = RNN_KAN.predict(cnd_model, cnd_rnn_x, cnd_input_scaler, cnd_output_scaler)
        p_cnd, h_ref_cnd_out = cnd_y_inv[0]
        
        P_cnd.append(p_cnd)
        H_ref_cnd_out.append(h_ref_cnd_out)

        h_cmp_out_array = h_cmp_out_array[1:]

        # evaporator subsystem
        # predict evaporator pressure and the outlet enthalpy
        evp_rnn_x = op_paras[['Tin','Tout','frequency','EEV']].values[-evp_K:,:]
        h_cnd_out_array = np.append(h_cnd_out_array, h_ref_cnd_out)
        evp_rnn_x = np.hstack((evp_rnn_x, h_cnd_out_array[-evp_K:].reshape(evp_K,1)))
        evp_y, evp_y_inv = RNN_KAN.predict(evp_model, evp_rnn_x, evp_input_scaler, evp_output_scaler)
        p_evp, h_ref_evp_out = evp_y_inv[0]

        P_evp.append(p_evp)
        H_ref_evp_out.append(h_ref_evp_out)

        h_cnd_out_array = h_cnd_out_array[1:]
        
        # compressor subsystem
        # predict compressor outlet enthalpy
        cmp_rnn_x = op_paras[['Tin','Tout','frequency','EEV']].values[-cmp_K:,:]
        h_evp_out_array = np.append(h_evp_out_array, h_ref_evp_out)
        p_evp_array = np.append(p_evp_array, p_evp)
        p_cnd_array = np.append(p_cnd_array, p_cnd)
        cmp_rnn_x = np.hstack((cmp_rnn_x, h_evp_out_array[-cmp_K:].reshape(cmp_K,1), p_evp_array[-cmp_K:].reshape(cmp_K,1), p_cnd_array[-cmp_K:].reshape(cmp_K,1)))
        cmp_y, cmp_y_inv = RNN_KAN.predict(cmp_model, cmp_rnn_x, cmp_input_scaler, cmp_output_scaler)
        h_ref_com_out = cmp_y_inv[0][0]

        h_evp_out_array = h_evp_out_array[1:]
        p_evp_array = p_evp_array[1:]
        p_cnd_array = p_cnd_array[1:]
        
    test_time = time.time()
    time_list.append(f'{i//PH}:Test time: {test_time-training_time}')
    
    # collect the actual data
    measured_p_cnd = data_for_test.iloc[i+K:i+K+PH,'p_cnd_colunm'].values
    measured_p_evp = data_for_test.iloc[i+K:i+K+PH,'p_evp_colunm'].values
    measured_h_evp_out = data_for_test.iloc[i+K:i+K+PH,'h_evp_colunm'].values
    measured_h_cnd_out = data_for_test.iloc[i+K:i+K+PH,'h_cnd_colunm'].values
    measured_h_cmp_out = data_for_test.iloc[i+K:i+K+PH,'h_cmp_colunm'].values
    
    # evaluate the current model
    mape_p_cnd, rmse_p_cnd = indicator(measured_p_cnd, P_cnd[i:i+PH])
    mape_p_evp, rmse_p_evp = indicator(measured_p_evp, P_evp[i:i+PH])
    mape_h_cmp, rmse_h_cmp = indicator(measured_h_cmp_out, H_ref_com_out[i:i+PH])
    mape_h_cnd, rmse_h_cnd = indicator(measured_h_cnd_out, H_ref_cnd_out[i:i+PH])
    mape_h_evp, rmse_h_evp = indicator(measured_h_evp_out, H_ref_evp_out[i:i+PH])
    
    indi_list.extend([
        mape_p_cnd, mape_p_evp, mape_h_cmp, mape_h_cnd, mape_h_evp,
        rmse_p_cnd, rmse_p_evp, rmse_h_cmp, rmse_h_cnd, rmse_h_evp
    ])
    
    # update the training set
    new_train_set = []
    for j in range(PH):
        if i+j+K+K+1 <= data_for_test.shape[0]:
            new_train_set.append(data_for_test[i+j+K:i+j+K+K])
        else:
            break
    new_train_set = np.array(new_train_set)
    new_train_set, _ = train_test_split(new_train_set, test_size=0.8, random_state=42)
    train_data_set = np.vstack((train_data_set, new_train_set))
    np.random.shuffle(train_data_set)
    train_data_set = train_data_set[new_train_set.shape[0]:,:,:]
    
    # determine if the update is needed
    start_time = time.time()
    if rmse_p_cnd >= rmse_p_cnd_b*tolerance:
        cnd_model, cnd_input_scaler, cnd_output_scaler = RNN_KAN.train(cnd_trainset_input, cnd_trainset_output, hidden_size[0], kan_hidden_size, grid_size)
    elif rmse_h_cnd >= rmse_h_cnd_b*tolerance:
        cnd_model, cnd_input_scaler, cnd_output_scaler = RNN_KAN.train(cnd_trainset_input, cnd_trainset_output, hidden_size[0], kan_hidden_size, grid_size)

    if rmse_p_evp >= rmse_p_evp_b*tolerance:
        evp_model, evp_input_scaler, evp_output_scaler = RNN_KAN.train(evp_trainset_input, evp_trainset_output, hidden_size[0], kan_hidden_size, grid_size)
    elif rmse_h_evp >= rmse_h_evp_b*tolerance:
        evp_model, evp_input_scaler, evp_output_scaler = RNN_KAN.train(evp_trainset_input, evp_trainset_output, hidden_size[0], kan_hidden_size, grid_size)

    if rmse_h_cmp >= rmse_h_cmp_b*tolerance:
        cmp_model, cmp_input_scaler, cmp_output_scaler = RNN_KAN.train(cmp_trainset_input, cmp_trainset_output, hidden_size[1], kan_hidden_size, grid_size)
    
    i += PH

# summarize the results
predict_data = list(zip(P_cnd, P_evp, H_ref_cnd_out, H_ref_evp_out, H_ref_com_out))
indi_list = np.array(indi_list).reshape(-1,10)
predict_df = pd.DataFrame(predict_data, index=data_for_test.index[K-1:K+len(P_cnd)-1], columns=['P_cnd', 'P_eap', 'H_cnd_out', 'H_evp_out', 'H_cmp_out'])
predict_df['delta_H_cnd'] = predict_df['H_cmp_out']-predict_df['H_cnd_out']

# save the results
predict_df.to_csv(f'{data_dir}/prediction_results_filename.csv')
with open(f'{data_dir}/prediction_results_filename.txt', 'w') as file:
    for item in time_list:
        file.write(f"{item}\n")
    for indi in indi_list:
        file.write(f"[{indi}]\n")