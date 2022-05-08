from leaf import Tensor

class Module(object):
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError(
        f'user defined nn.Module has not implemented forward pass')

    def parameters(self):
        # TODO: implement this
        return []

class Dense(Module):
    def __init__(self, fan_in, fan_out, use_bias=True):
        self.weight_ = Tensor.uniform(fan_in, fan_out)
        self.bias_ = Tensor.uniform(fan_out) if use_bias else None

    def forward(self, x):
        result = x.dot(self.weight_)
        
        if self.use_bias:
            return result.add(self.bias_)

        return result

class LSTM(Module):
    def __init__(self, input_size, hidden_size, n_layers):
        pass

    def forward(self, x, initial_states=None):
        batch_size, seq_len, item_len = x.shape
        hidden_seq = []
        HS = self.hidden_size

        if initial_states is None:
            h_t, c_t = (
                    Tensor.zeros(batch_size, HS, requires_grad=True),
                    Tensor.zeros(HS, HS, requires_grad=True))
        else: h_t, c_t = initial_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            gates = x_t @ self.W_ih.T + h_t @ self.W_hh.T + self.b_ih + self.b_hh

            i_t, f_t, g_t, o_t = 0,0,0,0


