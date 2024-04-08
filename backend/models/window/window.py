import numpy as np
import csv
import torch
from typing import List
from torch import nn, Tensor
from torch.optim import Adam

class Window(nn.Module):
    def __init__(self):
        super(Window, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
            nn.Linear(1, 1),
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
        for i in range(15_000):
            loss = self.fit(x, y)
            if i % 1000 == 0:
                print(loss)
        print(loss)
        return loss.item()
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        
    def predict(self, filename, data):
        model = self.load_model(filename)
        try:
            with torch.no_grad():
                prediction = model(data)
                return prediction
        except Exception as e:
            return f'Error: {e}'
    
    @classmethod
    def load_model(cls, filename):  # Change 'self' to 'cls'
        model = cls()  # Create an instance of the class
        model.load_state_dict(torch.load(filename))
        model.eval()
        return model
    
    
def data_to_csv(data: List[List[float]]) -> None:
    # Takes in a 2d array and writes the values to-
    # a csv file containing the header for the data.
    with open('sample.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
    
if __name__ == '__main__':
    
    data = Tensor([
        [0.7320644216691069],
        [0.636896046852123],
        [0.4985358711566618],
        [0.4494875549048316],
        
    ])
    
    sample_data = [
        ['Width', 'Output'], # Header for data
        [0.7320644216691069, 1],
        [0.636896046852123, 1],
        [0.4985358711566618, 0],
        [0.4494875549048316, 0],
    ]
    
    data_to_csv(sample_data)

    file = 'window2.pt'
    model = Window()
    # model.fit(new_x, new_y, epochs=15_000)
    # model.save_model(file)
    model = model.predict(file, data)
    print(model)
