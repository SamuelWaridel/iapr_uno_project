from matplotlib import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils.card_separation import CARD_W, CARD_H

def get_x_and_y(features, labels_df):
    zones = ["center", "player_1", "player_2", "player_3", "player_4"]

    data = []

    for img_id, feat in features.items():
        for zone in zones:
            labels = labels_df.loc[img_id].loc[zone]
            
            new_labels = []  
            labels = labels.split(";")
            for label in labels:
                if label[:2] in ["r_", "y_", "g_", "b_"]:
                    new_labels.append(label[2:])
                else:
                    new_labels.append(label)
            
            if len(new_labels) != 1:
                # skip if there is more than one card
                break
            else:
                for card in feat["cards"]:
                    if card["player"] == zone:
                        crop = card["crop"]
                        data.append((img_id, zone, crop, new_labels[0]))
                        break

    images = [sample[2] for sample in data]
    labels = [sample[3] for sample in data]
    
    return images, labels

label_to_idx = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "draw_2": 10,
    "skip": 11,
    "reverse": 12,
    "draw_4": 13,
    "wild": 14
}

idx_to_label = {
    v: k for k, v in label_to_idx.items()
}

class UnoDataset(Dataset):

    def __init__(self, images, labels, transform=None):

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        # Convert to numpy uint8
        image = np.array(image, dtype=np.uint8)

        # Optional debug
        # print(image.shape)

        # Convert to PIL
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert label string -> integer
        label = label_to_idx[label]

        # Convert label -> tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

class UnoCNN(nn.Module):

    def __init__(self, n_classes=15):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, n_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x
    
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    loss = running_loss / len(loader)
    acc = correct / total

    return loss, acc


def load_number_prediction_model(model_path, device):

    model = UnoCNN(n_classes=15)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def make_train_test_transforms():
    
    train_transform = transforms.Compose([
        transforms.Resize((CARD_W, CARD_H)),

        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((CARD_W, CARD_H)),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    return train_transform, test_transform