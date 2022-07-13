import math
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 前馈神经网络
class FnnLm(nn.Module):
    def __init__(self, n_class, m, n_step, n_hidden):
        super(FnnLm, self).__init__()
        self.n_step = n_step
        self.m = m
        self.C = nn.Embedding(n_class, m)
        self.w1 = nn.Linear(n_step * m, n_hidden, bias=False)
        self.b1 = nn.Parameter(torch.ones(n_hidden))
        self.w2 = nn.Linear(n_hidden, n_class, bias=False)
        self.w3 = nn.Linear(n_step * m, n_class, bias=False)

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, self.n_step * self.m)  # X
        Y1 = torch.tanh(self.b1 + self.w1(X))  # Y1 b1 w1
        b2 = self.w3(X)  # b2
        Y2 = b2 + self.w2(Y1)  # Y2
        return Y2


# Rnn调库
class RNN(nn.Module):
    def __init__(self, n_class, emb_size, n_hidden):
        super(RNN, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.rnn = nn.RNN(input_size=emb_size, hidden_size=n_hidden, num_layers=1)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embedding size]
        outputs, hidden = self.rnn(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1]  # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


class MyRnn(nn.Module):
    def __init__(self, n_class, emb_size, hidden_size):
        super(MyRnn, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        # embedding
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)  # (batch,seq_len,emb_size)
        # rnn without bias
        self.U = nn.Parameter(torch.empty(emb_size, hidden_size))
        self.V = nn.Parameter(torch.empty(hidden_size, hidden_size))
        # cls
        self.W = nn.Linear(hidden_size, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))
        # init weights
        self.init()

    def init(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, init_state=None):
        x = self.C(x)
        h_t = torch.zeros(x.shape[0], self.hidden_size).to(x.device)
        hidden_seq = []
        if init_state is not None:
            h_t = init_state
        for i in range(x.shape[1]):
            i_g = torch.matmul(x[:, i, :], self.U)
            h_g = torch.matmul(h_t, self.V)
            h_t = torch.tanh(i_g + h_g)
            hidden_seq.append(h_t)
        output = hidden_seq[-1]
        res = self.W(output) + self.b
        # return res, output, h_t
        return res


# 调库lstm
class Lstm(nn.Module):
    def __init__(self, n_class, emb_size, n_hidden):
        super(Lstm, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embedding size]
        outputs, hidden = self.lstm(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1]  # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


# 基于注意力机制的rnn
class Atten_RNN(nn.Module):
    def __init__(self, n_class, emb_size, n_hidden):
        super(Atten_RNN, self).__init__()
        self.n_hidden = n_hidden
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.rnn = nn.RNN(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(2 * n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embeding size]
        outputs, hidden = self.rnn(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        output = outputs[-1]
        attention = []
        for it in outputs[:-1]:
            attention.append(torch.mul(it, output).sum(dim=1).tolist())
        attention = torch.tensor(attention).to(device)
        attention = attention.transpose(0, 1)
        attention = torch.softmax(attention, dim=1).transpose(0, 1)
        # get soft attention
        attention_output = torch.zeros(outputs.size()[1], self.n_hidden).to(device)
        for i in range(outputs.size()[0] - 1):
            attention_output += torch.mul(attention[i], outputs[i].transpose(0, 1)).transpose(0, 1)
        output = torch.cat((attention_output, output), 1)
        # joint ouput output:[batch_size, 2*n_hidden]
        model = self.W(output) + self.b  # model : [batch_size, n_class]
        return model


# 基于注意力机制的Lstm
class Atten_Lstm(nn.Module):
    def __init__(self, n_class, emb_size, n_hidden):
        super(Atten_Lstm, self).__init__()
        self.n_hidden = n_hidden
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(2 * n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embeding size]
        outputs, hidden = self.lstm(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        output = outputs[-1]
        attention = []
        for it in outputs[:-1]:
            attention.append(torch.mul(it, output).sum(dim=1).tolist())
        attention = torch.tensor(attention).to(device)
        attention = attention.transpose(0, 1)
        attention = torch.softmax(attention, dim=1).transpose(0, 1)
        # get soft attention
        attention_output = torch.zeros(outputs.size()[1], self.n_hidden).to(device)
        for i in range(outputs.size()[0] - 1):
            attention_output += torch.mul(attention[i], outputs[i].transpose(0, 1)).transpose(0, 1)
        output = torch.cat((attention_output, output), 1)
        # joint ouput output:[batch_size, 2*n_hidden]
        model = self.W(output) + self.b  # model : [batch_size, n_class]
        return model
