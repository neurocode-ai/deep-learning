""" ---------------------------------------------------------------------------
Implementation of the Long Short Term Memory neural network. TODO: write more
docs.

Authors: Wilhelm Ågren <wilhelmagren98@gmail.com>
Last edited: 19-05-2022
License: Apache 2.0
--------------------------------------------------------------------------- """
from .nn import Module
    
class LSTM(Module):
    """ yeps this is bla bla bla

    Parameters
    ----------
    input_size: int
        Number of features in the input data, i.e., the length of each
        sequence element.
    hidden_size: int
        Number of features, neurons, to use for representing the hidden states.
    n_layers: int
        This does not matter currently, only one LSTM can be used for now.
    """
    def __init__(self, input_size, hidden_size, n_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.W_ih = Tensor.uniform(input_size, hidden_size*4, requires_grad=True)
        self.W_hh = Tensor.uniform(hidden_size, hidden_size*4, requires_grad=True)
        self.b_ih = Tensor.uniform(hidden_size*4, requires_grad=True)
        self.b_hh = Tensor.uniform(hidden_size*4, requires_grad=True)

    def forward(self, x, initial_states=None):
        batch_size, seq_len, item_len = x.shape
        hidden_seq = []
        HS = self.hidden_size

        if initial_states is None:
            h_t, c_t = (
                    Tensor.zeros(batch_size, HS, requires_grad=True),
                    Tensor.zeros(batch_size, HS, requires_grad=True))
        else: h_t, c_t = initial_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            gates = x_t.matmul(self.W_ih).add(h_t.matmul(self.W_hh)).add(self.b_ih.add(self.b_hh))
            
            i_t, f_t, g_t, o_t = gates.chunk(chunks=4, dim=1)
            i_t = i_t.sigmoid()
            f_t = f_t.sigmoid()
            g_t = g_t.tanh()
            o_t = o_t.sigmoid()
            c_t = f_t.multiply(c_t).add(i_t.multiply(g_t))
            h_t = o_t.multiply(c_t.tanh())
            hidden_seq.append(h_t)
        hidden_seq = leaf.concatenate(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)


