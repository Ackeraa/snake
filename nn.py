import torch
import torch.nn as nn
import random

class Net(nn.Module):   
    """Neural Network implementation.

    Attributes:
        a: Size of input layer.
        b: Size of hidden layer 1.
        c: Size of hidden layer 2.
        d: Size of output layer.
        fc1: Full connection layer 1.
        fc2: Full connection layer 2.
        out: Output layer.
        relu: Activation function of fc1 and fc2.
        sigmoid: Activation function of output.
    """    
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
        """Update the weights of the Neural Network."""
        weights = torch.FloatTensor(weights)
        with torch.no_grad():
            x = self.a * self.b
            xx = x + self.b
            y = xx + self.b * self.c
            yy = y + self.c
            z = yy + self.c * self.d
            self.fc1.weight.data = weights[0:x].reshape(self.b, self.a)
            self.fc1.bias.data = weights[x:xx]
            self.fc2.weight.data = weights[xx:y].reshape(self.c, self.b)
            self.fc2.bias.data = weights[y:yy]
            self.out.weight.data = weights[yy:z].reshape(self.d, self.c)
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
    model = Net(32, 20, 12, 4)

    weights = [random.random() for i in range(32 * 20 + 20 * 12 + 12 * 4 + 20 + 12 + 4)]
    model.update(weights)
    input = [random.random() for _ in range(32)]
    print(model.predict(input))
    model.show()
