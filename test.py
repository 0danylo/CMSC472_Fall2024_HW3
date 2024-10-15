import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Reset all random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class BirdVNonBirdDataset(Dataset):
    def __init__(self, file_name, mode='train'):
        self.file_name = file_name
        self.mode = mode
        data = torch.load(file_name)
        
        if self.mode == 'train':
            self.images = data['train_images']
            self.labels = data['train_labels']
        elif self.mode == 'val':
            self.images = data['val_images']
            self.labels = data['val_labels']
        elif self.mode == 'test':
            self.images = data['test_images']
        else:
            raise ValueError("Mode is not train, test, or val")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.mode == 'test':
            return image
        else:
            label = self.labels[idx]
            return image, label

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 7)
        self.fc2 = nn.Linear(7, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_one_epoch(model, train_dataloader, optimizer, loss_function):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Ensure labels are float and have the right shape
        labels = labels.float().view(-1, 1)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            print(f"Labels: {labels.view(-1).tolist()}")
            print(f"Outputs: {outputs.view(-1).tolist()}")
    
    return total_loss / len(train_dataloader)

def main():
    # Hyperparameters
    batch_size = 8
    learning_rate = 0.01
    epochs = 10
    input_size = 64 * 64 * 3

    # Dataset and DataLoader
    train_set = BirdVNonBirdDataset('toy_dataset.pth')
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Model, Optimizer, and Loss
    model = Net(input_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss()

    # Training Loop
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, loss_function)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()