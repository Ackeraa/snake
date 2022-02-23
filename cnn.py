import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Net(nn.Module):       
    def __init__(self, a, b, c):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3) #in_channel, out_channel, kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 15, 3)
        self.fc1 = nn.Linear(15, 20)
        self.fc2 = nn.Linear(20, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


    def update(self, weights):
        weights = torch.FloatTensor(weights)
        with torch.no_grad():
            x = 10 * 1 * 3 * 3
            xx = x + 10
            y = xx + 15 * 10 * 3 * 3
            yy = y + 15
            z = yy + 20 * 15
            zz = z + 20
            u = zz + 4 * 20
            self.conv1.weight.data = weights[0:x].reshape(10, 1, 3, 3)
            self.conv1.bias.data = weights[x:xx]
            self.conv2.weight.data = weights[xx:y].reshape(15, 10, 3, 3)
            self.conv2.bias.data = weights[y:yy]
            self.fc1.weight.data = weights[yy:z].reshape(20, 15)
            self.fc1.bias.data = weights[z:zz]
            self.fc2.weight.data = weights[zz:u].reshape(4, 20)
            self.fc2.bias.data = weights[u:]

    def predict(self, input):
        input = torch.tensor([[input]]).float()
        y = self(input)
        # print(y)
        return torch.argmax(y, dim=1).tolist()[0]

    def show(self):
        total_size = 0
        layer = 0
        with torch.no_grad():
            for parameters in self.parameters():
                layer += 1
                size = len(parameters.numpy().flatten())
                #print(parameters.numpy().flatten())
                total_size += size
        print("total:", total_size, layer)

        with torch.no_grad():
            print(self.conv1.weight.data.shape)
            print(self.conv1.bias.data.shape)
            print(self.conv2.weight.data.shape)
            print(self.conv2.bias.data.shape)
            print(self.fc1.weight.data.shape)
            print(self.fc1.bias.data.shape)
            print(self.fc2.weight.data.shape)
            print(self.fc2.bias.data.shape)
        

if __name__ == '__main__':
    w = 10
    h = 10
    model = Net(1, 2, 3)
    model.show()
    weights = [float(random.random()) for i in range(1869)]
    model.update(weights)
    model.show()
    input = [[random.random() for _ in range(w)] for _ in range(h)]
    print(model.predict(input))


