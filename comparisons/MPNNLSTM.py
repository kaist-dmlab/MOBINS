import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch_geometric.data import Data, Batch

class Config:
    def __init__(self, config_dict):
        self._config_dict = config_dict

    def __getattr__(self, item):
        return self._config_dict[item]    

""" GCN / RNN -Based Results
    Paper name: Transfer Graph Neural Networks for Pandemic Forecasting (AAAI 21) 
    Paper Implementation link(1): https://github.com/geopanag/pandemic_tgnn
    Additional Paper Implementation link(2): https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/mpnn_lstm.py
(MPNN, MPNNLSTM)
""" 

# data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

class Model(nn.Module):
    """
    MPNNLSTM
    """
    def __init__(self, opts):
        super(Model, self).__init__()
        configs = Config(opts)

        self.node_feature_dim = configs.node_feature_dim
        self.num_node = configs.num_node

        self.batch_size = opts["batch_size"]
        self.window = configs.seq_day * configs.cycle # self.seq_len 
        self.n_nodes = configs.num_node
        self.nhid = 64 # '--hidden', type=int, default=64,
        self.nfeat = (configs.num_node + configs.node_feature_dim)
        self.dropout = opts["dropout"]
        self.horizon = configs.pred_day * configs.cycle
        self.nout = self.horizon*(configs.num_node + configs.node_feature_dim)

        nfeat = self.nfeat
        nhid = self.nhid
        window = self.window
        
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear( nhid, self.nout)
        
        self.dropout = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.x # [B*W*num_nodes ,n_features] 
        edge_index = data.edge_index # [2,8160]
        edge_weight = torch.clamp(data.edge_weight, min=0) # [8160]
        #print(f"input x shape: {x.shape}") 
        #print(f"input edge_index shape: {edge_index.shape}")
        #print(f"input edge_weight shape: {edge_weight.shape}")
        assert not torch.isnan(data.x).any(), "NaN in input features."
        assert not torch.isnan(data.edge_weight).any(), "NaN in edge weights."
        
        channels = self.num_node*self.nfeat
        
        # Reshape and transpose the input for easier manipulation
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)
        batch_size = skip.shape[0]
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat)

        x = self.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        #print(f"after conv1: {x.shape}, {x}")
        assert not torch.isnan(x).any(), "NaN after conv1."
        x = self.bn1(x)
        x = self.dropout(x)
        lst = [x]
        
        x = self.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        x = torch.cat(lst, dim=1)
    
        x = x.view(batch_size, self.window, self.n_nodes, -1)
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))
    
        x, (hn1, cn1) = self.rnn1(x)
        out2, (hn2, cn2) = self.rnn2(x)
    
        x = torch.cat([hn1[0, :, :], hn2[0, :, :]], dim=1)
        skip = skip.reshape(skip.size(0), -1)
    
        x = torch.cat([x, skip], dim=1)
    
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
    
        x = x.reshape(batch_size, self.horizon, channels)
    
        return x