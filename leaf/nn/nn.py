from leaf import Tensor

class Module(object):
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError(
        f'user defined nn.Module has not implemented forward pass')

    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                if attr.requires_grad:
                    params.append(attr)
            if isinstance(attr, list):
                params.extend([p for p in attr if p.requires_grad])
            if isinstance(attr, Sequential):
                params.extend([p for p in attr.parameters() if p.requires_grad])
        return params

class Sequential(object):
    def __init__(self, *modules):
        self._modules = modules

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for m in self._modules:
            x = m(x)
        return x
    
    def parameters(self):
        params = []
        for m in self._modules:
            params.extend(m.parameters())
        return params
    
class Linear(Module):
    def __init__(self, fan_in, fan_out, use_bias=True):
        self.use_bias = use_bias
        self.weight_ = Tensor.uniform(fan_in, fan_out, requires_grad=True)
        self.bias_ = Tensor.uniform(fan_out, requires_grad=True) if use_bias else None

    def forward(self, x):
        result = x.matmul(self.weight_)
        
        if self.use_bias:
            return result.add(self.bias_)

        return result

class LSTM(Module):
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
                    Tensor.zeros(HS, HS, requires_grad=True))
        else: h_t, c_t = initial_states

        for t in range(seq_len):
            # TODO: IMPLEMENT ARRAY ACCESSING FOR TENSOR, __getitem__
            x_t = x[:, t, :]
            gates = x_t.matmul(self.W_ih.T).add(h_t.matmul(self.W_hh.T)).add(self.b_ih.add(self.b_hh))

            i_t, f_t, g_t, o_t = gates.chunk(chunks=4, dim=1)
            i_t = i_t.sigmoid()
            f_t = f_t.sigmoid()
            g_t = g_t.tanh()
            o_t = o_t.sigmoid()
            c_t = f_t.multiply(c_t).add(i_t.multiply(g_t))
            h_t = o_t.multiply(c_t.tanh())
            hidden_seq.append(h_t)
        print(hidden_seq)

class LogSoftmax(Module):
    def forward(self, x):
        return x.logsoftmax()

class ReLU(Module):
    def forward(self, x):
        return x.relu()
