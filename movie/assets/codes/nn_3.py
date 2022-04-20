import torch
import torch.nn as nn

class Net(nn.Module):   

    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, weights):
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

        self.set_weights(weights)













if __name__ == '__main__':

    weights = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
    nn = Net(1, 2, 1, 2, weights)
    print(nn.predict([1]))
#########################################################################
