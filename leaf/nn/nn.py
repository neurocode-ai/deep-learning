import leaf
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
                params.extend([attr] if attr.requires_grad else [])
            if isinstance(attr, list):
                params.extend([p for p in attr if p.requires_grad])
            if isinstance(attr, (Module, Sequential)):
                params.extend([p for p in attr.parameters() if p.requires_grad])
        return params

class Sequential(object):
    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x
    
    def parameters(self):
        params = []
        for m in self.modules:
            params.extend(m.parameters())
        return params
    
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

class Tanh(Module):
    def forward(self, x):
        return x.tanh()

class LogSoftmax(Module):
    def forward(self, x):
        return x.logsoftmax()

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

