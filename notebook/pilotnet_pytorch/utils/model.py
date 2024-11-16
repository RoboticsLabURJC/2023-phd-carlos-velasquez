import torch
import torch.nn as nn
import torch.optim as optim

class PilotNetTwoOutput(nn.Module):
    def __init__(self, img_shape=(66, 200, 4), dropout=0.1):
        super(PilotNetTwoOutput, self).__init__()
        
        # Capa de normalización para 4 canales de entrada
        self.batch_norm = nn.BatchNorm2d(4, eps=0.001)
        
        # Capas convolucionales
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        
        # Capas completamente conectadas
        self.fc1 = nn.Linear(64 * 1 * 18, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        
        # Dos salidas: steer y throttle
        self.fc5_steer = nn.Linear(10, 1)
        self.fc5_throttle = nn.Linear(10, 1)
        
        # Dropout y activación
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Normalización por lotes
        x = self.batch_norm(x)
        
        # Pasar por las capas convolucionales
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        
        # Aplanar
        x = x.view(x.size(0), -1)
        
        # Pasar por las capas completamente conectadas con dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        
        # Dos salidas separadas para steer y throttle
        steer_output = self.fc5_steer(x)
        throttle_output = self.fc5_throttle(x)
        
        return steer_output, throttle_output

# Inicialización del modelo con dos salidas
def initialize_model(img_shape=(66, 200, 4), learning_rate=0.0001, dropout=0.1):
    model = PilotNetTwoOutput(img_shape=img_shape, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Puedes usar MSE para cada salida individual

    return model, optimizer, criterion
