import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
from models import MyModel, MyModel2,VGG,ResNet18

# 检查cuda是否可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
class MyDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.classes = sorted(self.df[1].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid, label = self.df.iloc[index, :]
        img_list = os.listdir(os.path.join(self.root, f"{vid}"))
        img_list = sorted(img_list)
        img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
 
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes.index(label)
        return img, label


transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


trainval_dataset = MyDataset("/home/chenhuil/hkustgz-aiaa-5032-hw2-spring-2024/video_frames_30fpv_320p", "/home/chenhuil/hkustgz-aiaa-5032-hw2-spring-2024/labels/trainval.csv", transform)
train_data, val_data = train_test_split(trainval_dataset, test_size=0.2, random_state=0)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

net = VGG().to(device)
# net = ResNet18().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
patience = 5
best_acc = 0
counter = 0  # Initialize counter to track epochs since last improvement

for epoch in range(50):
    start_time = time.time() # 记录开始时间
    running_loss_train = 0.0
    running_loss_val = 0.0
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0
    
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # Training code ...
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss_train = criterion(outputs, labels)
        loss_train.backward()
        optimizer.step()
        running_loss_train += loss_train.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            # Validation code ...
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = net(val_inputs)
            loss_val = criterion(val_outputs, val_labels)
            running_loss_val += loss_val.item()
            _, predicted_val = torch.max(val_outputs.data, 1)
            total_val += val_labels.size(0)
            correct_val += (predicted_val == val_labels).sum().item()

    acc_val = correct_val/len(val_loader)
    if acc_val > best_acc:
        best_acc = acc_val
        counter = 0  # Reset counter
        torch.save(net.state_dict(), '/home/chenhuil/hw/hw2/VGG16_model_best.pth')
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping: validation accuracy did not improve for {} epochs'.format(patience))
            break  # Stop training loop

    torch.save(net.state_dict(), '/home/chenhuil/hw/hw2/VGG16_last.pth')
    end_time = time.time() # 记录结束时间
    print('[Epoch %d] Loss (train/val): %.3f/%.3f' % (epoch + 1, running_loss_train/len(train_loader), running_loss_val/len(val_loader)),
          ' Acc (train/val): %.2f%%/%.2f%%' % (100 * correct_train/total_train, 100 * correct_val/total_val)
          ,' Epoch Time: %.2f' % (end_time - start_time))
