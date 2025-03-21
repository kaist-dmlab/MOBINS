import numpy as np
import pandas as pd
import os

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from scipy.optimize import linprog

def linear_regression_fit(n, x):
    y = - 1 / (n+1) * (x-1) + 1
    return y

def exponential_regression_fit(n, x):
    y = np.exp(-(x-1))
    return y

def k_hop_matrix(adjacency_matrix, max_hops):
    # Initialize the k-hop matrix with the original adjacency matrix
    k_hop_matrix = np.array(adjacency_matrix, dtype=np.float32)
    
    # Raise the adjacency matrix to the power of 2 up to max_hops
    for _ in range(2, max_hops + 1):
        k_hop_matrix += np.linalg.matrix_power(adjacency_matrix, _) # Reachability matrix
    
    # Set values greater than max_hops to 0
    k_hop_matrix[k_hop_matrix > max_hops] = 0
    
    # Replace the diagonal with 1s from the identity matrix
    np.fill_diagonal(k_hop_matrix, 1)
    
    return k_hop_matrix

# Generated training sequences for use in the model.
def _create_sequences(values, len_lookback, len_forecast, stride=1):
    tensor_x = []
    tensor_y = []
    values = torch.from_numpy(values).float()
    for i in range(0, len(values) - len_lookback - len_forecast + 1, stride):
        tensor_x.append(values[i : i + len_lookback])
        tensor_y.append(values[i + len_lookback : i + len_lookback + len_forecast])
   
    return torch.stack(tensor_x), torch.stack(tensor_y)

def _reshape_for_normalization(value):
    days, timestamp, dim = value.size()
    value_2d = value.reshape(days*timestamp, dim)
    return value_2d

def _reshape_for_restore(value, cycle):
    hours_per_day = cycle
    days_time, dim = value.shape[0], value.shape[1]
    days = days_time // hours_per_day
    value = value.reshape(days, hours_per_day, dim)
    return value

def _get_time_features(start_date, end_date):
    start_date = start_date
    end_date = end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Make dataframe for time features
    df_stamp = pd.DataFrame()
    df_stamp['date'] = pd.to_datetime(date_range.date)
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
    df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
    df_stamp = df_stamp.drop(columns=['date']).values
    return df_stamp

def _make_windowing_and_loader(dataset, model, batch_size, tensor, seq_day, pred_day, test_ratio, train_ratio, cycle, opts):
    # Define the lookback window and forecasting window
    lookback_window = seq_day  # Number of days to look back
    forecasting_window = pred_day  # Number of days to forecast
    test_start_index = len(tensor) - int(len(tensor) * test_ratio) # Assume you want to use the last 90 days for testing

    # train, test seperate
    ts_train = tensor[:test_start_index]
    ts_test = tensor[test_start_index:]
    
    ts_train = _reshape_for_normalization(ts_train)
    ts_test = _reshape_for_normalization(ts_test)

    scaler = StandardScaler()
    
    ts_train = scaler.fit_transform(ts_train)
    ts_test = scaler.transform(ts_test)

    scaled_ts_train = _reshape_for_restore(ts_train, cycle)
    scaled_ts_test = _reshape_for_restore(ts_test, cycle)
    
    ts_train_x, ts_train_y = _create_sequences(scaled_ts_train, lookback_window, forecasting_window)
    # print("TS TrainX: ", ts_train_x.shape, "TS TrainY: ", ts_train_y.shape)
    ts_test_x, ts_test_y = _create_sequences(scaled_ts_test, lookback_window, forecasting_window)
    # print("TS TestX: ", ts_test_x.shape, "TS TestY: ", ts_test_y.shape)

    ts_train_x_reshaped = ts_train_x.reshape(ts_train_x.size(0), ts_train_x.size(1)*ts_train_x.size(2), ts_train_x.size(3))
    ts_train_y_reshaped = ts_train_y.reshape(ts_train_y.size(0), ts_train_y.size(1)*ts_train_y.size(2), ts_train_y.size(3))
    ts_test_x_reshaped = ts_test_x.reshape(ts_test_x.size(0), ts_test_x.size(1)*ts_test_x.size(2), ts_test_x.size(3))
    ts_test_y_reshaped = ts_test_y.reshape(ts_test_y.size(0), ts_test_y.size(1)*ts_test_y.size(2), ts_test_y.size(3))

    # Split the data into training and validation sets
    train_size = int(train_ratio * len(ts_train_x_reshaped))
    train_input = ts_train_x_reshaped[:train_size]
    train_target = ts_train_y_reshaped[:train_size]
    
    val_input = ts_train_x_reshaped[train_size:]
    val_target = ts_train_y_reshaped[train_size:]

    if model in ['Autoformer', 'TimesNet', 'Informer', 'Reformer','D2STGNN']:
        if dataset == 'nyc': 
            start_date = '2022-02-01' 
            end_date = '2024-03-31' 
        elif dataset == 'korea_covid':
            start_date = '2020-01-20'
            end_date = '2023-08-31'
        elif dataset == 'nyc_covid':
            start_date = '2020-03-01'
            end_date = '2023-12-31'
        elif dataset == 'busan':
            start_date = '2021-01-01'
            end_date = '2023-12-31'
        elif dataset == 'daegu':
            start_date = '2021-01-01'
            end_date = '2023-12-31'              
        elif dataset == 'seoul':
            start_date = '2022-01-01'
            end_date = '2023-12-31'
        data_stamp = _get_time_features(start_date, end_date)

        seq_train_mark = data_stamp[:test_start_index]
        seq_test_mark = data_stamp[test_start_index:]    
        seq_train_mark = scaler.fit_transform(seq_train_mark)
        seq_test_mark = scaler.transform(seq_test_mark)        
        train_x_mark, train_y_mark = _create_sequences(seq_train_mark, lookback_window, forecasting_window)
        test_x_mark, test_y_mark = _create_sequences(seq_test_mark, lookback_window, forecasting_window)

        train_x_mark = train_x_mark.repeat(1, cycle, 1)
        train_y_mark = train_y_mark.repeat(1, cycle, 1)
        test_x_mark = test_x_mark.repeat(1, cycle, 1)
        test_y_mark = test_y_mark.repeat(1, cycle, 1)

        train_input_mark = train_x_mark[:train_size]
        train_target_mark = train_y_mark[:train_size]
        val_input_mark = train_x_mark[train_size:]
        val_target_mark = train_y_mark[train_size:]
            
        train_dataset = TensorDataset(train_input, train_target, train_input_mark, train_target_mark)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(val_input, val_target, val_input_mark, val_target_mark)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(ts_test_x_reshaped, ts_test_y_reshaped, test_x_mark, test_y_mark)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    elif model in ['MPNNLSTM']:
        train_dataset = TensorDataset(train_input, train_target)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, opts,"train"))
        val_dataset = TensorDataset(val_input, val_target)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, opts,"val"))
        test_dataset = TensorDataset(ts_test_x_reshaped)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: create_pyg_data(batch, opts,"test"))
    else:
        train_dataset = TensorDataset(train_input, train_target)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(val_input, val_target)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(ts_test_x_reshaped)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader, ts_test_y



def create_pyg_data(x_batch, opts,mode): 
    window = opts["seq_day"] * opts["cycle"]
    num_nodes = opts["num_node"]
    node_feature_dim = opts["node_feature_dim"]    
    pyg_data_list = []

    if mode=="test": 
        x_batch_list = [x_tensor[0] for x_tensor in x_batch]
        x_batch = torch.cat(x_batch_list, dim=0) # [B*window,n*n_features]
        x_batch = x_batch.reshape(-1,window,num_nodes,node_feature_dim + num_nodes)
    node_features = x_batch.reshape(-1, num_nodes, node_feature_dim + num_nodes) # [B*window,n,n_features]
    adj = node_features[:, :, node_feature_dim:].reshape(-1, num_nodes, num_nodes)
    #node_features = node_features[:, :, :node_feature_dim]

    for t in range(node_features.shape[0]):
        x_t = node_features[t,:,:]
        adj_t = adj[t]

        # Get edge_index and edge_weight from adj_t
        edge_index = adj_t.nonzero(as_tuple=False).t().contiguous()
        edge_weight = adj_t[edge_index[0], edge_index[1]]

        # Create PyG Data object
        data = Data(x=x_t, edge_index=edge_index, edge_weight=edge_weight)
        pyg_data_list.append(data)
    
    # Create a batch of graphs
    pyg_batch = Batch.from_data_list(pyg_data_list)
    return pyg_batch

def collate_fn(batch, opts,mode):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    pyg_batch = create_pyg_data(inputs, opts,mode)
    return pyg_batch, targets


def _make_loader(batch_size, ts_latent_tensor, od_latent_tensor, complex_tensor, seq_day, pred_day, test_ratio, cycle, train_ratio):
    # Define the lookback window and forecasting window
    lookback_window = seq_day  # Number of days to look back
    forecasting_window = pred_day  # Number of days to forecast
    test_start_index = len(ts_latent_tensor) - int(len(ts_latent_tensor) * test_ratio)
    
    # train, test seperate
    ts_train = ts_latent_tensor[:test_start_index]
    ts_test = ts_latent_tensor[test_start_index:]
    od_train = od_latent_tensor[:test_start_index]
    od_test = od_latent_tensor[test_start_index:]
    complex_train = complex_tensor[:test_start_index]
    complex_test = complex_tensor[test_start_index:]
    
    ts_train = _reshape_for_normalization(ts_train)
    ts_test = _reshape_for_normalization(ts_test)
    od_train = _reshape_for_normalization(od_train)
    od_test = _reshape_for_normalization(od_test)
    complex_train = _reshape_for_normalization(complex_train)
    complex_test = _reshape_for_normalization(complex_test)

    scaler = StandardScaler()
    
    ts_train = scaler.fit_transform(ts_train)
    ts_test = scaler.transform(ts_test)
    od_train = scaler.fit_transform(od_train)
    od_test = scaler.transform(od_test)
    complex_train = scaler.fit_transform(complex_train)
    complex_test = scaler.transform(complex_test)    

    scaled_ts_train = _reshape_for_restore(ts_train, cycle)
    scaled_ts_test = _reshape_for_restore(ts_test, cycle)
    scaled_od_train = _reshape_for_restore(od_train, cycle)
    scaled_od_test = _reshape_for_restore(od_test, cycle)
    scaled_complex_train = _reshape_for_restore(complex_train, cycle)
    scaled_complex_test = _reshape_for_restore(complex_test, cycle)
    
    ts_train_x, ts_train_y = _create_sequences(scaled_ts_train, lookback_window, forecasting_window)
    od_train_x, od_train_y = _create_sequences(scaled_od_train, lookback_window, forecasting_window)
    complex_train_x, complex_train_y = _create_sequences(scaled_complex_train, lookback_window, forecasting_window)
    
    ts_test_x, ts_test_y = _create_sequences(scaled_ts_test, lookback_window, forecasting_window)
    od_test_x, od_test_y = _create_sequences(scaled_od_test, lookback_window, forecasting_window)
    complex_test_x, complex_test_y = _create_sequences(scaled_complex_test, lookback_window, forecasting_window)


    # Create Validation Dataset
    train_size = int(train_ratio * len(complex_train_x))
    timeseries_train_x, timeseries_train_y = ts_train_x[:train_size], ts_train_y[:train_size]
    origindestin_train_x, origindestin_train_y = od_train_x[:train_size], od_train_y[:train_size]
    com_train_x, com_train_y = complex_train_x[:train_size], complex_train_y[:train_size]
    
    ts_val_x, ts_val_y = ts_train_x[train_size:], ts_train_y[train_size:]
    od_val_x, od_val_y = od_train_x[train_size:], od_train_y[train_size:]
    com_valid_x, com_valid_y = complex_train_x[train_size:], complex_train_y[train_size:]

    # Create a PyTorch Dataset
    train_dataset = TensorDataset(timeseries_train_x, origindestin_train_x, com_train_x, com_train_y)

    # Create a PyTorch Dataset
    test_dataset = TensorDataset(ts_test_x, od_test_x, complex_test_x, complex_test_y)
    
    # Create a PyTorch Dataset
    valid_dataset = TensorDataset(ts_val_x, od_val_x, com_valid_x, com_valid_y)

    # Create test DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, valid_loader, complex_train_y, complex_test_y


def load_datasets(dataset, khop=0, only_adj=False, ar_adj=False,opts=None):
    # Dataset directory
    ROOT_PATH = './dataset/'  # CHANGE YOUR DIR 
    # Various dataset
    if dataset == "seoul": DATASET_MODE = 'Transportation-Seoul'
    elif dataset == "busan": DATASET_MODE = 'Transportation-Busan'
    elif dataset == "daegu": DATASET_MODE = 'Transportation-Daegu'
    elif dataset == "nyc": DATASET_MODE = 'Transportation-NYC'
    elif dataset == "korea_covid": DATASET_MODE = 'Epidemic-Korea'
    elif dataset == "nyc_covid": DATASET_MODE = 'Epidemic-NYC'

    # Final path
    TS_DIR= ROOT_PATH + DATASET_MODE+'/NODE_TIME_SERIES_FEATURES/'
    OD_DIR= ROOT_PATH + DATASET_MODE+'/OD_MOVEMENTS/'
    ADJ_FILE = ROOT_PATH + DATASET_MODE+'/SPATIAL_NETWORK.npy'
    
    # Topology Data
    graph_adjacency_matrix = torch.tensor(np.load(ADJ_FILE), dtype=torch.float32)
    if khop > 0:
        original_matrix = graph_adjacency_matrix
        for hop in range(1, khop+1):
            k_hop_matrix_result = k_hop_matrix(original_matrix, hop)
            continuous_value = exponential_regression_fit(khop, hop)
            R_per_hop = continuous_value*(k_hop_matrix_result>0)
            graph_adjacency_matrix += continuous_value*((original_matrix==0) & (R_per_hop>0))  
    if only_adj==True:
        return graph_adjacency_matrix
        
    # Origin-Destination Data
    od_datasets  = sorted([f for f in os.listdir(f'{OD_DIR}') if os.path.isfile(os.path.join(f'{OD_DIR}', f))])
    od_stacks = []
    for npy in od_datasets:
        node_feature_data = torch.tensor(np.load(OD_DIR + npy), dtype=torch.float32)
        node_feature_data = np.transpose(np.array(node_feature_data), (2, 0, 1))
        od_stacks.append(node_feature_data)
    od_tensors = torch.Tensor(np.stack(od_stacks))
    
    # Node Feature TS Data
    ts_datasets  = sorted([f for f in os.listdir(f'{TS_DIR}') if os.path.isfile(os.path.join(f'{TS_DIR}', f))])

    ts_stacks = []
    for npy in ts_datasets:
        time_series_data = torch.tensor(np.load(TS_DIR + npy), dtype=torch.float32)
        time_series_data = np.transpose(np.array(time_series_data), (1, 0, 2))
        ts_stacks.append(time_series_data)
    ts_tensors = torch.tensor(np.stack(ts_stacks))

    ts_tensors_3d = ts_tensors.reshape(ts_tensors.size(0), ts_tensors.size(1), ts_tensors.size(2)*ts_tensors.size(3))
    od_tensors_3d = od_tensors.reshape(od_tensors.size(0), od_tensors.size(1), od_tensors.size(2)*od_tensors.size(3))
    
    # Multi-Modal Concat (High-dim)
    complex_3d = torch.cat((ts_tensors_3d, od_tensors_3d), dim=-1)
    return ts_tensors_3d, od_tensors_3d, complex_3d, graph_adjacency_matrix


def wasserstein_distance(p, q, D):
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = np.array(D)
    D = D.reshape(-1)

    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    myresult = result.fun

    return myresult

def spatial_temporal_aware_distance(x, y):
    x, y = np.array(x), np.array(y)
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    return wasserstein_distance(p, q, D)


def spatial_temporal_similarity(x, y, normal, transpose):
    if normal:
        x = normalize(x)
        x = normalize(x)
    if transpose:
        x = np.transpose(x)
        y = np.transpose(y)
    return 1 - spatial_temporal_aware_distance(x, y)

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std


