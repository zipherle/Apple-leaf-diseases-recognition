import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
from datetime import datetime
from sklearn.metrics import accuracy_score
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
root_dir = "datasets/"
train_data = datasets.ImageFolder(root_dir, transform=transform)
print([classes for classes in train_data.classes])  # ['APPLE ROT LEAVES', 'HEALTHY LEAVES', 'LEAF BLOTCH', 'SCAB LEAVES']

trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))  # Adjust final layer to match number of classes
num_epochs = 2200
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00075)
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for X, y in trainloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            _, preds = torch.max(y_pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Epoch: {epoch}/{num_epochs}, Loss: {running_loss/len(trainloader):.4f},Accuracy: {accuracy * 100:.2f}% at {datetime.now()}')
torch.save(model.state_dict(), "model/cnn-model3.pth")
