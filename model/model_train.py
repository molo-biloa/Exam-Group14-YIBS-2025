import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import kagglehub

# Download Fashion-MNIST dataset
path = kagglehub.dataset_download("zalando-research/fashionmnist")

# Enhanced data transformations with normalization and augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),      # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load dataset using PyTorch
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Convert to numpy arrays for visualization
train_images = train_data.data.numpy()
train_labels = train_data.targets.numpy()

# Explore dataset
print(f"Training data shape: {train_images.shape}")  # (60000, 28, 28)
print(f"Test data shape: {test_data.data.shape}")    # (10000, 28, 28)

# Plot sample images
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(class_names[train_labels[i]])
plt.show()

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)  # Increased batch size
test_loader = DataLoader(test_data, batch_size=128, num_workers=4)  # Increased batch size

# Enhanced CNN model with BatchNorm and deeper layers
class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Keep spatial dimensions
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample to 14x14

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample to 7x7

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample to 3x3

            # Classifier
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),  # Increased hidden units
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)  # Learning rate scheduler

# Training loop with detailed logging
epochs = 15  # Increased epochs
train_losses = []  # Store training losses
val_losses = []    # Store validation losses

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training phase
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 20)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Print every 100 batches
        if (batch_idx + 1) % 100 == 0:
            current_loss = running_loss / total
            current_acc = 100 * correct / total
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                  f"Loss: {current_loss:.4f} | "
                  f"Accuracy: {current_acc:.2f}%")

    # Store training loss
    epoch_loss = running_loss / total
    train_losses.append(epoch_loss)

    # Epoch statistics
    epoch_acc = 100 * correct / total
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Training Loss: {epoch_loss:.4f} | "
          f"Training Accuracy: {epoch_acc:.2f}%")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    # Store validation loss
    val_loss = val_loss / val_total
    val_losses.append(val_loss)

    val_acc = 100 * val_correct / val_total
    print(f"  Validation Loss: {val_loss:.4f} | "
          f"Validation Accuracy: {val_acc:.2f}%")
    print("-" * 50)

    # Learning rate scheduler step
    scheduler.step(val_loss)

# Generate loss chart after training
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, epochs+1))
plt.legend()
plt.grid(True)
plt.show()

# Final evaluation
model.eval()
final_correct = 0
final_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        final_total += labels.size(0)
        final_correct += (predicted == labels).sum().item()

print("\nFinal Test Results:")
print("-" * 50)
print(f"Test Accuracy: {100 * final_correct / final_total:.2f}%")

# Save model
torch.save(model.state_dict(), 'fashion_mnist_cnn_v3.pth')
print("\nModel saved successfully!")