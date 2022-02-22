import torch
import torch.nn as nn
import random

class Net(nn.Module):       
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()

        self.a = n_input
        self.b = n_hidden1
        self.c = n_hidden2
        self.d = n_output

        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, n_output)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.out(y)
        y = self.sigmoid(y)
        return y

    def update(self, weights):
        with torch.no_grad():
            for i in range(len(weights)):
                weights[i] = torch.FloatTensor(weights[i])
            self.fc1.weight.data = weights[0].reshape(self.b, self.a)
            self.fc2.weight.data = weights[1].reshape(self.c, self.b)
            self.out.weight.data = weights[2].reshape(self.d, self.c)
            self.fc1.bias.data = weights[3]
            self.fc2.bias.data = weights[4]
            self.out.bias.data = weights[5]
        
            '''
            x = self.a * self.b
            y = x + self.b * self.c
            z = y + self.c * self.d
            xx = z + self.b
            yy = xx + self.c
            zz = yy + self.d
            self.fc1.weight.data = weights[0 : x].reshape(self.b, self.a)
            self.fc2.weight.data = weights[x : y].reshape(self.c, self.b)
            self.out.weight.data = weights[y : z].reshape(self.d, self.c)
            self.fc1.bias.data = weights[z : xx]
            self.fc2.bias.data = weights[xx : yy]
            self.out.bias.data = weights[yy : zz]
            '''
        # self.show()

    def predict(self, input):
        input = torch.tensor([input]).float()
        y = self(input)
        return torch.argmax(y, dim=1).tolist()[0]

    def show(self):
        for parameters in self.parameters():
            print(parameters)

if __name__ == '__main__':
    model = Net(32, 20, 12, 4)
    f = open('in.txt', 'r')
    lines = f.readlines()
    weights = [ float(lines[i]) for i in range(32 * 20 + 20 * 12 + 12 * 4 + 20 + 12 + 4)]
    model.update(weights)
    input = [random.random() for _ in range(32)]
    print(model.predict(input))
    #model.show()
