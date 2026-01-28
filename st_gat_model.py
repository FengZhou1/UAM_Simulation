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
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(STGAT, self).__init__()
        self.dropout = dropout

        # Spatial Graph Attention
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        
        # Temporal Processing (LSTM)
        # Input to LSTM: (batch, seq_len, num_nodes * nhid) -> Flatten nodes? 
        # Or shared LSTM per node? Let's do shared LSTM per node for scalability.
        self.lstm = nn.LSTM(input_size=nhid, hidden_size=nhid, num_layers=1, batch_first=True)
        
        # Output prediction
        self.linear = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        # x: (batch, seq_len, num_nodes, nfeat)
        batch_size, seq_len, num_nodes, nfeat = x.size()
        
        # Process each time step through GAT
        # Flatten batch and seq_len for spatial processing
        x_reshaped = x.view(batch_size * seq_len, num_nodes, nfeat) #(B*T, N, F)
        
        # Apply GAT
        x_gat = torch.cat([att(x_reshaped, adj) for att in self.attentions], dim=2)
        x_gat = F.dropout(x_gat, self.dropout, training=self.training)
        x_gat = self.out_att(x_gat, adj) # (B*T, N, nhid)
        x_gat = F.elu(x_gat)
        
        # Reshape for Temporal
        # We want to treat each node as a sequence
        # (B, T, N, H) -> (B*N, T, H)
        nhid = x_gat.size(-1)
        x_seq = x_gat.view(batch_size, seq_len, num_nodes, nhid)
        x_seq = x_seq.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, seq_len, nhid)
        
        # LSTM
        lstm_out, _ = self.lstm(x_seq)
        # Take last time step
        last_out = lstm_out[:, -1, :] # (B*N, H)
        
        # Prediction
        out = self.linear(last_out) # (B*N, nclass)
        out = out.view(batch_size, num_nodes, nclass)
        
        return out
