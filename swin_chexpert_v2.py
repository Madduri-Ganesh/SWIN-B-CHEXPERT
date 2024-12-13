# %%
import torch

# %%
#Data Loader

import os
import csv
import random, copy 
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm  # Pretrained models like Swin Transformer
import numpy as np
from PIL import Image
# %%
import csv
import pandas as pd
from sklearn.metrics import roc_auc_score
import time

print(torch.cuda.is_available())

class CheXpertDataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        # print(label)
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        # print(label)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


# %%
class CheXpertTestDataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[1:]
        # print(label)
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        # print(label)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)

# %%
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# %%
train_dataset = CheXpertDataset(images_path='~/dataset/CheXpert-v1.0',file_path='~/dataset/CheXpert-v1.0/CheXpert-v1.0/train.csv', augment=transform, num_class=14)
test_dataset = CheXpertTestDataset(images_path='~/dataset/CheXpert-v1.0',file_path='~/CheXpert-v1.0/CheXpert-v1.0/test.csv', augment=transform,  num_class=14)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=10)

data = pd.read_csv('/data/jliang12/jpang12/dataset/CheXpert-v1.0/CheXpert-v1.0/test.csv')
print(data.head())




# Define basic device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model (Swin Transformer - base version)
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=14)  # CheXpert is a 5-class problem
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # CheXpert is a multi-label classification task
optimizer = optim.AdamW(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# %%


# Function for testing the model
def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.BCEWithLogitsLoss()  # Use appropriate loss function
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculation for testing
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Store predictions and true labels for AUC calculation
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    # Compute average loss
    avg_loss = running_loss / len(test_loader)

    # Flatten lists for AUC calculation
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC-ROC for each class
    aucs = []
    for i in range(all_labels.shape[1]):
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        aucs.append(auc)

    return avg_loss, aucs


# Example of using the testing function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming `test_dataset` and `model` are already defined
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Test the model
test_loss, test_aucs = test_model(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")
for idx, auc in enumerate(test_aucs):
    print(f"Class {idx+1} AUC: {auc:.4f}")

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    start = time.time()
    # print(f"The time for 1 epoch is {start:.2f} seconds")
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    final = time.time()
    print(f"The time for {epoch} epoch is {final:.2f} seconds")
    test_loss, test_aucs = test_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Update the scheduler
    scheduler.step(test_loss)

print("Training completed!")


# %%


# Function for testing the model
def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.BCEWithLogitsLoss()  # Use appropriate loss function
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculation for testing
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Store predictions and true labels for AUC calculation
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    # Compute average loss
    avg_loss = running_loss / len(test_loader)

    # Flatten lists for AUC calculation
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC-ROC for each class
    aucs = []
    for i in range(all_labels.shape[1]):
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        aucs.append(auc)

    return avg_loss, aucs


# Example of using the testing function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming `test_dataset` and `model` are already defined
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Test the model
test_loss, test_aucs = test_model(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")
for idx, auc in enumerate(test_aucs):
    print(f"Class {idx+1} AUC: {auc:.4f}")


# %%



