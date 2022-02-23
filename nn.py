import torch
import torch.nn as nn
import random

class Net(nn.Module):       
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()

        self.a = n_input
        self.b = n_hidden
        self.c = n_output

        self.fc = nn.Linear(n_input, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        y = self.relu(y)
        y = self.out(y)
        y = self.sigmoid(y)
        return y

    def update(self, weights):
        weights = torch.FloatTensor(weights)
        with torch.no_grad():
            x = self.a * self.b
            y = x + self.b
            z = y + self.b * self.c
            self.fc.weight.data = weights[0:x].reshape(self.b, self.a)
            self.fc.bias.data = weights[x:y]
            self.out.weight.data = weights[y:z].reshape(self.c, self.b)
            self.out.bias.data = weights[z:]

    def predict(self, input):
        input = torch.tensor([input]).float()
        y = self(input)
        return torch.argmax(y, dim=1).tolist()[0]

    def show(self):
        with torch.no_grad():
            for parameters in self.parameters():
                print(parameters.numpy().flatten())

if __name__ == '__main__':
    model = Net(32, 20, 4)

    weights = [random.random() for i in range(32 * 20 + 20 * 4 + 20 + 4)]
    model.update(weights)
    input = [random.random() for _ in range(32)]
    print(model.predict(input))
    model.show()
