import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams["font.sans-serif"]=["Times New Roman"]
plt.rcParams["axes.unicode_minus"]=False
from sklearn.model_selection import train_test_split
import RNN_KAN
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import time
import warnings
warnings.filterwarnings("ignore")

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
df = pd.read_csv(data_dir+'/data/1T_test_data.csv',encoding='gb2312')
df['MCGS_TIME'] = pd.to_datetime(df['MCGS_TIME'])
data_for_test = df.set_index('MCGS_TIME')

cnd_K = 4
evp_K = 6
cmp_K = 7

K = max(cnd_K, evp_K, cmp_K)
train_data_set = np.load(f'{data_dir}/data/KAN/temp_data/training_data_1T_K={K}.npy', allow_pickle=True)
train_data_set, _  = train_test_split(train_data_set, test_size=0.8, random_state=42)

time_list = []
hidden_size = (30,40)
kan_hidden_size = 20
grid_size = 10
# training model
cmp_trainset_input = train_data_set[:, :, [0,1,2,3,7,5,4]].astype(float)
cmp_trainset_output = train_data_set[:, -1, [9]].astype(float)
cnd_trainset_input = train_data_set[:, :, [0,1,2,3,6]].astype(float)
cnd_trainset_output = train_data_set[:, -1, [4,8]].astype(float)
evp_trainset_input = train_data_set[:, :, [0,1,2,3,8]].astype(float)
evp_trainset_output = train_data_set[:, -1, [5,7]].astype(float)

start_time = time.time()
cmp_model, cmp_input_scaler, cmp_output_scaler = RNN_KAN.train(cmp_trainset_input, cmp_trainset_output, hidden_size[1], kan_hidden_size, grid_size)
cnd_model, cnd_input_scaler, cnd_output_scaler = RNN_KAN.train(cnd_trainset_input, cnd_trainset_output, hidden_size[0], kan_hidden_size, grid_size)
evp_model, evp_input_scaler, evp_output_scaler = RNN_KAN.train(evp_trainset_input, evp_trainset_output, hidden_size[0], kan_hidden_size, grid_size)

training_time = time.time()
time_list.append(f'Training time: {training_time-start_time}')
N = data_for_test.shape[0]
P_cnd, P_evp, H_ref_cnd_out, H_ref_evp_out, H_ref_com_out = [], [], [], [], []

# initialize
h_cmp_out_array = data_for_test.iloc[0:K-1,6].values
h_cnd_out_array = data_for_test.iloc[0:K-1,8].values
h_evp_out_array = data_for_test.iloc[0:K-1,7].values
p_cnd_array = data_for_test.iloc[0:K-1,4].values
p_evp_array = data_for_test.iloc[0:K-1,5].values

# for condenser as starting subsystem
h_ref_com_out = data_for_test.iloc[K-1,6]

for t in range(N-K):
    op_paras = data_for_test.iloc[t:t+K,0:4]
    H_ref_com_out.append(h_ref_com_out)

    # condenser subsystem
    # predict condenser pressure and the outlet enthalpy
    cnd_rnn_x = op_paras[['Tin','室外温度','压缩机运行频率','电子膨胀阀步数']].values[-cnd_K:]
    h_cmp_out_array = np.append(h_cmp_out_array,h_ref_com_out)
    cnd_rnn_x = np.hstack((cnd_rnn_x, h_cmp_out_array[-cnd_K:].reshape(cnd_K,1)))
    cnd_y, cnd_y_inv = RNN_KAN.predict(cnd_model, cnd_rnn_x, cnd_input_scaler, cnd_output_scaler)
    p_cnd, h_ref_cnd_out = cnd_y_inv[0]

    h_cmp_out_array = h_cmp_out_array[1:]
    
    P_cnd.append(p_cnd)
    H_ref_cnd_out.append(h_ref_cnd_out)

    # evaporator subsystem
    # predict evaporator pressure and the outlet enthalpy
    evp_rnn_x = op_paras[['Tin','室外温度','压缩机运行频率','电子膨胀阀步数']].values[-evp_K:]
    h_cnd_out_array = np.append(h_cnd_out_array, h_ref_cnd_out)
    evp_rnn_x = np.hstack((evp_rnn_x, h_cnd_out_array[-evp_K:].reshape(evp_K,1)))
    evp_y, evp_y_inv = RNN_KAN.predict(evp_model, evp_rnn_x, evp_input_scaler, evp_output_scaler)
    p_evp, h_ref_evp_out = evp_y_inv[0]
    
    P_evp.append(p_evp)
    H_ref_evp_out.append(h_ref_evp_out)
    
    h_cnd_out_array = h_cnd_out_array[1:]
    
    # compressor subsystem
    # predict compressor outlet enthalpy
    cmp_rnn_x = op_paras[['Tin','室外温度','压缩机运行频率','电子膨胀阀步数']].values[-cmp_K:]
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
time_list.append(f'Test time: {test_time-training_time}')
time_list.append(f'Total time: {test_time-start_time}')

# summarize the results
predict_data = list(zip(P_cnd, P_evp, H_ref_com_out, H_ref_evp_out, H_ref_cnd_out))
predict_df = pd.DataFrame(predict_data, index=data_for_test.index[K-1:-1], columns=['P_cnd', 'P_evp','H_cmp_out','H_evp_out','H_cnd_out'])
predict_df['delta_H_cnd'] = predict_df['H_cmp_out']-predict_df['H_cnd_out']
merged_df = pd.merge(predict_df, data_for_test.iloc[:,4:-1], left_index=True, right_index=True)
merged_df['delta_h_cnd'] = merged_df['h_cmp_out'] - merged_df['h_cnd_out']

# save the results
merged_df.to_csv(f'{data_dir}/data/KAN/batch_training/cnd_hybrid.csv', encoding='gb2312')
with open(f'{data_dir}/data/KAN/batch_training/cnd.txt', 'w') as file:
    for item in time_list:
        file.write(f"{item}\n")