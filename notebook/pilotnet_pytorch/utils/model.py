import torch
import torch.nn as nn
import torch.optim as optim

class PilotNet(nn.Module):
    def __init__(self, img_shape=(66, 200, 4), dropout=0.3):
        super(PilotNet, self).__init__()
        
        
        self.batch_norm = nn.BatchNorm2d(4, eps=0.001)
        
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        
        
        self.fc1 = nn.Linear(64 * 1 * 18, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)
        
        
        self.dropout = nn.Dropout(p=dropout)
        
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.batch_norm(x)
        
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        
        
        x = x.view(x.size(0), -1)
        
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        
        
        x = self.fc5(x)
        
        return x


def initialize_model(img_shape=(66, 200, 4), learning_rate=0.0001, dropout=0.1):
    model = PilotNet(img_shape=img_shape, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  

    return model, optimizer, criterion

