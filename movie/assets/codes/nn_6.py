import torch
import torch.nn as nn

class Net(nn.Module):   

    def predict(self, input):
        input = torch.tensor([input]).float()
        y = self(input)
        return torch.argmax(y, dim=1).tolist()[0]
































#########################################################################
