# Necessary Library Import

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# Set up the device-GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


class ResNetViT(nn.Module):
    def __init__(self, num_classes=2, num_transformer_layers=1, num_heads=8, embed_dim=256):
        super(ResNetViT, self).__init__()

        # Load pretrained ResNet-50 and modify to extract feature maps
        self.resnet = models.resnet50(pretrained=True)

        # Remove the final fully connected layer to extract only the feature maps
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Projection layer to reduce ResNet-50 output dimension from 2048 to embed_dim (256)
        self.projection = nn.Linear(2048, embed_dim)

        # Transformer Encoder Configuration
        transformer_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2048,
                                                    dropout=0.1)
        self.transformer = TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # Fully Connected Classification Layer for binary classification
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Step 1: Pass through ResNet-50 for feature extraction
        features = self.resnet(x)  # Shape: (batch_size, 2048, 7, 7)
        batch_size, num_channels, height, width = features.shape

        # Step 2: Flatten the feature map to prepare it for transformer
        features = features.view(batch_size, num_channels, height * width)  # Shape: (batch_size, 2048, 49)
        features = features.permute(0, 2, 1)  # Shape: (batch_size, 49, 2048)

        # Step 3: Apply linear projection to match the transformer's embedding size
        features = self.projection(features)  # Shape: (batch_size, 49, 256)

        # Step 4: Apply transformer to learn global dependencies
        transformer_output = self.transformer(features)  # Shape: (batch_size, 49, 256)

        # Step 5: Global average pooling on the transformer output
        transformer_output = transformer_output.mean(dim=1)  # Shape: (batch size, 256)

        # Step 6: Classification
        out = self.fc(transformer_output)  # Shape: (batch_size, 2)
        return out


# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset (adjust paths as needed)
train_dataset = datasets.ImageFolder("/Train Dataset Path", transform=transform)
val_dataset = datasets.ImageFolder("/Validation Dataset Path", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize the model, move it to the GPU if available
model = ResNetViT(embed_dim=256, num_transformer_layers=1).to(device)

# Use Cross Entropy Loss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Use SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)  # Get predicted class
    return accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())


# Training and Validation Loop with CUDA and metric tracking
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    # Lists to store accuracy and loss values for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (torch.argmax(outputs, 1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (torch.argmax(outputs, 1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Store the losses and accuracies for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Print epoch metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

    # Plotting the learning curves
    plt.figure(figsize=(12, 6))

    # Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# Train and validate the model
train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

# Test Case


# Define the test data transformations (same as training and validation)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test dataset
data_dir = '\Test Dataset'  # Update this path
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transforms)

# Create DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# Check class labels assigned by ImageFolder
print(test_dataset.class_to_idx)

# Check the number of images in the test dataset
print(f"Number of images in the test dataset: {len(test_dataset)}")

# Check the number of batches in the test DataLoader
print(f"Number of batches in the test DataLoader: {len(test_loader)}")

from sklearn.metrics import precision_score, recall_score, f1_score


# Function to evaluate model on test data
def evaluate_test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation during inference
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # Using global device variable
            labels = labels.to(device)

            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predictions

            test_correct += (preds == labels).sum().item()  # Count correct predictions
            test_total += labels.size(0)  # Count total samples

            all_preds.extend(preds.cpu().numpy())  # Store all predictions
            all_labels.extend(labels.cpu().numpy())  # Store all true labels

    # Calculate overall test accuracy
    test_accuracy = test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # Calculate additional metrics
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')

    # Print the computed metrics
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')

    return all_preds, all_labels, test_accuracy, precision, recall, f1



# Run the test evaluation
all_preds, all_labels, test_accuracy, test_precision, test_recall, test_f1 = evaluate_test_model(model, test_loader)








