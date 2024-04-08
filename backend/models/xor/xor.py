import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        self.optimizer = Adam(self.parameters())
        self.loss = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
    def fit(self, x, y):
        self.optimizer.zero_grad()
        y_prediction = self.forward(x)
        loss = self.loss(y, y_prediction)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    
if __name__ == '__main__':
    import time
    x = Tensor(np.array([
        [0, 0], [0, 1],
        [1, 0], [1, 1],
        [0, 0], [0, 1],
        [1, 0], [1, 1],
        [0, 0], [0, 1],
        [1, 0], [1, 1],
    ]))
    
    y = Tensor(np.array([
        [0], [1],
        [1], [0],
        [0], [1],
        [1], [0],
        [0], [1],
        [1], [0],
    ]))
    
    model = XORModel()
    
    for i in range(10_000):
        loss = model.fit(x, y)
        if i % 1000 == 0:
            print(loss)
    print(loss)
    
    print(
        model(Tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ])).round()
    )
