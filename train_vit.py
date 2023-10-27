from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import wandb
from pytorch_pretrained_vit import ViT

import argparse
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint file for resuming training")
args = parser.parse_args()

batch_size = 32
epochs = 500
lr = 3e-5
num_classes = 225

wandb.init(project="AUTSL100_colorData_ViT", name='AUTSL_all_data_pretrained_384', config={
           "learning_rate": lr})
#desired_image_size = (128, 128)

data_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    #    transforms.Lambda(lambda x: x.unsqueeze(0)),
    transforms.Normalize(0.5, 0.5),
])

# Tensor Transforms (with Augmentation) and Pytorch Preprocessing:
train_ds = torchvision.datasets.ImageFolder(
    "../AUTSL_converted_data/train", transform=data_transforms)
valid_ds = torchvision.datasets.ImageFolder(
    "../AUTSL_converted_data/val", transform=data_transforms)
test_ds = torchvision.datasets.ImageFolder(
    "../AUTSL_converted_data/test", transform=data_transforms)

# Data Loaders:
train_loader = data.DataLoader(train_ds, batch_size=batch_size, num_workers=4)
valid_loader = data.DataLoader(valid_ds, batch_size=batch_size, num_workers=4)
test_loader = data.DataLoader(test_ds, batch_size=batch_size, num_workers=4)

# Training device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomHead(nn.Module):
    def __init__(self, num_classes):
        super(CustomHead, self).__init__()
        # Assuming the ViT model outputs 768 features
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


epoch = 0
if args.resume is not None:
    model = ViT('B_16_imagenet1k')
    model.head = CustomHead(num_classes)
    modelPath = args.resume
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Loading model from {args.resume}.")
    #model = nn.DataParallel(model)
    #checkpoint = torch.load(modelPath)
    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

else:
    print("not resuming")
    model = ViT('B_16_imagenet1k', pretrained=True)
    model.head = CustomHead(num_classes)

model.eval()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_val_loss = float('inf')

for epoch in range(epoch, epochs):
    print("number of epochs: ", epoch)
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
            bestModelPath = 'weights/best_all_data.pth'
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss}, bestModelPath)
            else:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss}, bestModelPath)

        lastModelPath = 'weights/last_all_data.pth'
        if torch.cuda.device_count() > 1:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss}, lastModelPath)
        else:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss}, lastModelPath)
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
        test_loss += loss_t.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_loss = test_loss/len(test_loader)
print("Testing Results:")
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {(100 * correct / total):.2f}%")
#torch.save(model.state_dict(), 'best_model.pth')
