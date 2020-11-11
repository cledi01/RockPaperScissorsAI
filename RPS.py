import random
from model import *
import numpy as np

def player(my_play,prev_play,my_history=[],opponent_history=[]):
    RPSNet = NeuralNet()
    RPSNet.load_state_dict(torch.load('model.ckpt'))
    if (prev_play != ''):
        opponent_history.append(prev_play)
        my_history.append(my_play)
    plays = ["R","P","S"]
    playDict = {"R":0,"P":1,"S":2}
    winDict = {"R":1,"P":2,"S":0}
    if len(opponent_history)<50:
        guess = random.randint(0,2)

    else:
        arr = [playDict[move] for move in opponent_history[-50:]]
        arr.extend([playDict[move] for move in my_history[-50:]])
        tensor = torch.tensor(arr,dtype=torch.float)
        # tensor.unsqueeze_(0)
        _, guess = predict(RPSNet,tensor)

    if (len(opponent_history)>=100):
        arr = [playDict[move] for move in opponent_history[-51:-1]]
        arr.extend(playDict[move] for move in my_history[-51:-1])
        tensor = torch.tensor(arr,dtype=torch.float)
        # tensor.unsqueeze_(0)

        output, _ = predict(RPSNet,tensor)
        train(RPSNet,
            output.unsqueeze(0),
            torch.tensor(winDict[opponent_history[-1]]).unsqueeze(0))

    torch.save(RPSNet.state_dict(),'model.ckpt')
    return plays[guess]
