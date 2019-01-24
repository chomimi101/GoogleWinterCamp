# coding=utf-8

import torch
from torch import nn
from torch.autograd import Variable
from utils import runBiRNN
from models.modules.Attention import Attention
from models.modules.GlobalAttention import GlobalAttention


class PointerNetRNNDecoder(nn.Module):
    """
    Pointer network RNN Decoder, process all the output together
    """
    def __init__(self, args, input_dim):
        super(PointerNetRNNDecoder, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(input_dim, self.args.hidden_size, batch_first=False, bidirectional=True,
                                  num_layers=self.args.num_layers, dropout=self.args.dropout_p)
        self.attention = GlobalAttention(args=self.args, dim=2 * self.args.hidden_size, attn_type="mlp")
        
    def forward(self, tgt, src, hidden, src_lengths=None, src_max_len=None, tgt_lengths=None, tgt_max_len=None):
        def _fix_enc_hidden(h):
            """
            The encoder hidden is  (layers*directions) x batch x dim.
            We need to convert it to layers x batch x (directions*dim).
            """
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h
        
        # RNN
        # rnn_output, hidden = self.lstm(tgt, hidden)
        rnn_output, hidden = runBiRNN(self.lstm, inputs=tgt, seq_lengths=tgt_lengths, hidden=hidden, total_length=tgt_max_len)
        # Attention
        rnn_output = rnn_output.transpose(0, 1).contiguous()
        # _, (batch_size, tgt, src)
        attn_h, align_score = self.attention(input=rnn_output, context=src, context_lengths=src_lengths, context_max_len=src_max_len)
        return align_score, rnn_output, hidden
