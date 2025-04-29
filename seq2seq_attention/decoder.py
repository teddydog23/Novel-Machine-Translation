# model/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq_attention.attention import Attention

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # input: [batch_size]
        embedded = self.embed(input).unsqueeze(0)  # [1, batch, embed_size]
        embedded = self.dropout(embedded)

        # Attention
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)  # [batch, 1, src_len]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))      # [batch, 1, hidden]
        context = context.transpose(0, 1)                                # [1, batch, hidden]

        rnn_input = torch.cat([embedded, context], 2)                    # [1, batch, embed+hidden]
        output, hidden = self.gru(rnn_input, last_hidden)

        output = output.squeeze(0)  # [batch, hidden]
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))  # [batch, output_size]
        # output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights
