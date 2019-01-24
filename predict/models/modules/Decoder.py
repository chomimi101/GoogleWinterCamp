# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
from models.modules.Attention import Attention


class AttnDecoderRNN(nn.Module):
    def __init__(self, args):
        super(AttnDecoderRNN, self).__init__()
        self.args = args
        self.linear = nn.Linear(self.args.word_dim, 2*self.args.hidden_size)
        self.attn = nn.Linear(self.args.hidden_size * 4, self.args.tokenize_max_len)
        self.attn_combine = nn.Linear(self.args.hidden_size * 4, self.args.hidden_size)
        self.lstm = nn.LSTM(args.word_dim, args.hidden_size, bidirectional=True, batch_first=False,
                            num_layers=args.num_layers, dropout=args.dropout_p)
        self.attention = Attention('general', dim=2 * self.args.hidden_size, args=self.args)

    def forward(self, input, hidden, encoder_outputs, src_lengths=None, src_max_len=None):
        embedded = self.linear(input).unsqueeze(1)
        # (B, 1, 2H)
        hidden_b = self._cat_directions(hidden[0]).transpose(0, 1)
        # (B, 1, 2H)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden_b), 2)), dim=2)
        # (B, 1, L)
        attn_applied = attn_weights.matmul(encoder_outputs)
        # (B, 1, 2H)

        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)
        output = F.relu(output)
        # (B, 1, H)
        output, hidden = self.lstm(output, hidden)

        attn_h, align_score = self.attention(encoder_outputs, output, src_lengths=src_lengths, src_max_len=src_max_len)
        return align_score, hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
