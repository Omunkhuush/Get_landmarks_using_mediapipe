from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from vit_pytorch.efficient import ViT
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import torch.utils.data as data
import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import wandb
from pytorch_pretrained_vit import ViT

torch.cuda.empty_cache()

batch_size = 64
epochs = 500
lr = 3e-5
gamma = 0.7
seed = 142
IMG_SIZE = 512
patch_size = 8
num_classes = 225

wandb.init(project="AUTSL100_colorData_ViT", name='AUTSL_all_data_different_color_pretrained_384', config={
           "learning_rate": lr})
#desired_image_size = (128, 128) 

data_transforms = transforms.Compose([
    transforms.Resize((384, 384)), 
    transforms.ToTensor(),
#    transforms.Lambda(lambda x: x.unsqueeze(0)), 
    transforms.Normalize(0.5, 0.5),
])

# Tensor Transforms (with Augmentation) and Pytorch Preprocessing:
train_ds = torchvision.datasets.ImageFolder("../AUTSL_full_diff_color_all_hand_with_pose3/train", transform=data_transforms)
valid_ds = torchvision.datasets.ImageFolder("../AUTSL_full_diff_color_all_hand_with_pose3/val", transform=data_transforms)
test_ds = torchvision.datasets.ImageFolder("../AUTSL_full_diff_color_all_hand_with_pose3/test", transform=data_transforms)

# Data Loaders:
train_loader = data.DataLoader(train_ds, batch_size=batch_size, num_workers=4)
valid_loader = data.DataLoader(valid_ds, batch_size=batch_size, num_workers=4)
test_loader  = data.DataLoader(test_ds, batch_size=batch_size, num_workers=4)

# Training device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomHead(nn.Module):
    def __init__(self, num_classes):
        super(CustomHead, self).__init__()
        self.fc = nn.Linear(768, num_classes)  # Assuming the ViT model outputs 768 features

    def forward(self, x):
        x = self.fc(x)
        return x
model = ViT('B_16_imagenet1k', pretrained=True)
model.head = CustomHead(num_classes)
model.eval()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data, label in tqdm(train_loader):
        inputs = data.to(device)
        labels = label.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        val_loss = 0.0
        print('Validing...\n')
        for data, label in valid_loader:
            #inputs, labels = batch
            inputs = data.to(device)
            labels = label.to(device)
            #inputs = inputs.contiguous()
            outputs = model(inputs)

            loss_v = criterion(outputs, labels)
            val_loss += loss_v.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(valid_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save the model's state_dict to a file
            #torch.save(model.state_dict(), 'best_model.pth')
            #torch.save(model.state_dict(), f'weights/best_{epoch}.pth')
            torch.save(model.state_dict(), 'weights/best_all_data.pth')
        torch.save(model.state_dict(), 'weights/last_all_data.pth')
        wandb.log({"Training Loss": train_loss,
                   "Validation Loss": val_loss,
                   "Validation Accuracy": (100 * correct / total)})

    print(f"Training Loss: {train_loss:.4f}\n")
    print(f"Validation Loss: {val_loss:.4f}\n")
    print(f"Validation Accuracy: {(100 * correct / total):.4f}%\n")

# Testing loop
model.eval()
with torch.no_grad():
    test_loss = 0.0
    correct = 0
    total = 0
    for data, label in test_loader:
        #inputs, labels = batch
        inputs = data.to(device)
        labels = label.to(device)
        outputs = model(inputs)
        loss_t = criterion(outputs, labels)
        test_loss+=loss_t.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_loss = test_loss/len(test_loader)
print("Testing Results:")
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {(100 * correct / total):.2f}%")
#torch.save(model.state_dict(), 'best_model.pth')
