import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the hyperparameters
config = dict(
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=0.005,
    dataset="MNIST",
    architecture="CNN"
)

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, config.kernels[0], kernel_size=3)
        self.conv2 = nn.Conv2d(config.kernels[0], config.kernels[1], kernel_size=3)
        self.fc1 = nn.Linear(config.kernels[1] * 24 * 24, 128)
        self.fc2 = nn.Linear(128, config.classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 24 * 24 * config['kernels'][1])
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to create model, data, optimizer, and loss function
def make(config):
    model = SimpleCNN(config)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer

# Train the model
def train(model, train_loader, criterion, optimizer, config):
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        wandb.log({"epoch": epoch, "loss": total_loss / len(train_loader)})

# Test the model
def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    wandb.log({"accuracy": accuracy})
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Main pipeline function
def model_pipeline(hyperparameters):
    with wandb.init(project="mnist-classification", config=hyperparameters):
        config = wandb.config

        model, train_loader, test_loader, criterion, optimizer = make(config)

        # Train and evaluate
        train(model, train_loader, criterion, optimizer, config)
        test(model, test_loader)

    return model

# Run the pipeline
if __name__ == "__main__":
    model_pipeline(config)
