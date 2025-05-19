# model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden]
        # encoder_outputs: [seq_len, batch, hidden]

        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)  # [batch, seq_len, hidden]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch, seq_len, hidden]

        attn_energies = self.score(h, encoder_outputs)  # [batch, seq_len]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # [batch, 1, seq_len]

    def score(self, hidden, encoder_outputs):
        # Concatenate hidden and encoder outputs: [batch, seq_len, 2*hidden]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [batch, seq_len, hidden]
        energy = energy.transpose(1, 2)  # [batch, hidden, seq_len]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [batch, 1, hidden]
        energy = torch.bmm(v, energy)  # [batch, 1, seq_len]
        return energy.squeeze(1)  # [batch, seq_len]
