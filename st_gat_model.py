import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: (batch_size, num_nodes, in_features)
        # adj: (num_nodes, num_nodes) 
        # Note: simplistic implementation, usually GAT handles sparse adj
        
        batch_size = h.size(0)
        Wh = torch.matmul(h, self.W) # (batch, N, out)
        
        # Attention mechanism
        # Prepare for broadcasting to compute all pairs (N, N)
        # We want to compute a[Wh_i || Wh_j]
        
        # a_input: (batch, N, N, 2*out)
        N = Wh.size(1)
        a_input = torch.cat([Wh.repeat(1, 1, N).view(batch_size, N * N, -1),
                             Wh.repeat(1, N, 1)], dim=2).view(batch_size, N, N, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # (batch, N, N)

        zero_vec = -9e15*torch.ones_like(e)
        # Mask using adj
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class STGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n_nodes, dropout, alpha, nheads):
        super(STGAT, self).__init__()
        self.dropout = dropout
        self.n_nodes = n_nodes
        self.nhid = nhid

        # Spatial Graph Attention
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        
        # Temporal Processing (Centralized LSTM as per ref)
        # Represents the global state evolution
        # Input: Flattened state of all nodes [N * nhid] per time step
        self.lstm_input_size = n_nodes * nhid
        self.lstm_hidden_size = 128
        
        self.lstm1 = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.lstm_hidden_size, hidden_size=256, num_layers=1, batch_first=True)
        
        # Output prediction
        # Map back from global hidden state to per-node prediction
        self.linear = nn.Linear(256, n_nodes * nclass)

    def forward(self, x, adj):
        # x: (batch, seq_len, num_nodes, nfeat)
        batch_size, seq_len, num_nodes, nfeat = x.size()
        
        assert num_nodes == self.n_nodes, f"Input node count {num_nodes} does not match model node count {self.n_nodes}"
        
        # Process each time step through GAT
        # Flatten batch and seq_len for spatial processing
        x_reshaped = x.view(batch_size * seq_len, num_nodes, nfeat) #(B*T, N, F)
        
        # Apply GAT
        x_gat = torch.cat([att(x_reshaped, adj) for att in self.attentions], dim=2)
        x_gat = F.dropout(x_gat, self.dropout, training=self.training)
        x_gat = self.out_att(x_gat, adj) # (B*T, N, nhid)
        x_gat = F.elu(x_gat)
        
        # Reshape for Centralized LSTM
        # (B*T, N, nhid) -> (B, T, N, nhid) -> (B, T, N*nhid)
        x_seq = x_gat.view(batch_size, seq_len, num_nodes, self.nhid)
        x_flat = x_seq.view(batch_size, seq_len, num_nodes * self.nhid)
        
        # LSTM
        # (B, T, InputSize)
        out1, _ = self.lstm1(x_flat)
        out2, _ = self.lstm2(out1)
        
        # Take last time step
        last_out = out2[:, -1, :] # (B, Hidden2)
        
        # Prediction
        out = self.linear(last_out) # (B, N*nclass)
        
        # Reshape to (Batch, N, nclass)
        n_class = out.size(1) // num_nodes
        out = out.view(batch_size, num_nodes, n_class)
        
        return out
