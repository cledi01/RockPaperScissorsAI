import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(100,100)
        self.fc2 = nn.Linear(100,100)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(100,3)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def predict(net,last10plays):
    prediction = net(last10plays)
    predictedMove = np.argmax(prediction.detach().numpy())
    # Returns the index of the predicted move
    return (prediction,predictedMove)

def train(net,output,win_move):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.0001)
    # Forward pass
    loss = criterion(output,win_move)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# test = NeuralNet()
# torch.save(test.state_dict(),'model.ckpt')