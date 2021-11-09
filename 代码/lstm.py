import torch
import torch.nn as nn
#模型构建
class LSTM(torch.nn.Module):
    def __init__(self,input_size,hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #LSTM主体部分
        self.linearix = torch.nn.Linear(self.input_size, self.hidden_size)
        self.linearfx = torch.nn.Linear(self.input_size, self.hidden_size)
        self.lineargx = torch.nn.Linear(self.input_size, self.hidden_size)
        self.linearox = torch.nn.Linear(self.input_size, self.hidden_size)
        self.linearih = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.linearfh = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.lineargh = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.linearoh = torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, hidden, c):
        i = self.sigmoid(self.linearix(x) + self.linearih(hidden))
        f = self.sigmoid(self.linearfx(x) + self.linearfh(hidden))
        g = self.tanh(self.lineargx(x) + self.lineargh(hidden))
        o = self.sigmoid(self.linearox(x) + self.linearoh(hidden))
        c = i * c + f * g
        hidden = o * self.tanh(c)
        return hidden, c
#初始化参数
input_size = 2
seq_size = 5
hidden_size= 6
batch_size = 1
#输出模型结果
x = torch.ones(seq_size,batch_size,input_size)
hidden = torch.ones(batch_size,hidden_size)
c = torch.ones(batch_size,hidden_size)
net = LSTM(input_size,hidden_size)
net(x,hidden,c)