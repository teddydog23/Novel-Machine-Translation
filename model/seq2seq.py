# model/seq2seq.py

import torch
import torch.nn as nn
import random
from torch.autograd import Variable

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [seq_len, batch]
        # trg: [seq_len, batch]
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(src.device)

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]  # Match layers

        input = trg[0, :]  # <sos>

        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs
