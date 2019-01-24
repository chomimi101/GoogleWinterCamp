import torch
import torch.nn as nn
from utils import aeq


class GlobalAttention(nn.Module):
    """
    Luong Attention.
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.
        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                      a
    Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.
    Luong Attention (dot, general):
    The full function is
    $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.
    * dot: $$score(h_t,{\overline{h}}_s) = h_t^T{\overline{h}}_s$$
    * general: $$score(h_t,{\overline{h}}_s) = h_t^T W_a {\overline{h}}_s$$
    Bahdanau Attention (mlp):
    $$c = \sum_{j=1}^{SeqLength}\a_jh_j$$.
    The Alignment-function $$a$$ computes an alignment as:
    $$a_j = softmax(v_a^T \tanh(W_a q + U_a h_j) )$$.
    """

    def __init__(self, args, dim, is_transform_out=True, attn_type="mlp", attn_hidden=0):
        """
        :param attn_hidden: whether use self.transform_in for src and tgt
        """
        super(GlobalAttention, self).__init__()

        self.args = args
        self.dim = dim
        self.attn_type = attn_type
        self.attn_hidden = attn_hidden
        assert (self.attn_type in ["dot", "general", "mlp"]), (
            "Please select a valid attention type.")

        if attn_hidden > 0:
            self.transform_in = nn.Sequential(
                nn.Linear(dim, attn_hidden),
                nn.ELU(0.1))

        if self.attn_type == "general":
            d = attn_hidden if attn_hidden > 0 else dim
            self.linear_in = nn.Linear(d, d, bias=False)
            # initialization
            # self.linear_in.weight.data.add_(torch.eye(d))
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        if is_transform_out:
            self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)
        else:
            self.linear_out = None

        self.tanh = nn.Tanh()

    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x tgt_len x dim
        h_s (FloatTensor): batch x src_len x dim
        returns scores (FloatTensor): batch x tgt_len x src_len:
            raw attention scores for each src index
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_hidden > 0:
                h_t = self.transform_in(h_t)
                h_s = self.transform_in(h_s)
            if self.attn_type == "general":
                h_t = self.linear_in(h_t)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, input, context, context_lengths=None, context_max_len=None):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output.
        context (FloatTensor): batch x src_len x dim: src hidden states
        """

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # compute attention scores, as in Luong et al.
        align = self.score(input, context)

        if context_lengths is not None:
            mask = self.sequence_mask(context_lengths, context_max_len)
            mask = mask.unsqueeze(1)
            # (bz, max_len) -> (bz, 1, max_len), so mask can broadcast
            align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = torch.softmax(align, dim=-1)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)
        # concatenate
        concat_c = torch.cat([c, input], -1)
        # linear_out
        if self.linear_out is None:
            attn_h = concat_c
        else:
            attn_h = self.linear_out(concat_c)
            if self.attn_type in ["general", "dot"]:
                attn_h = self.tanh(attn_h)
        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        # (batch, targetL, dim_), (batch, targetL, sourceL)
        return attn_h, align_vectors

    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return torch.arange(0, max_len).to(lengths.device).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1))
