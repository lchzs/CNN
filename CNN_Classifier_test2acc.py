import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from models import MyModel, MyModel2,VGG,ResNet18

# 检查cuda是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        img_list = os.listdir(os.path.join(self.root, f"{vid}.mp4"))
        img_list = sorted(img_list)
        img_path = os.path.join(self.root, f"{vid}.mp4", img_list[int(len(img_list)/2)])
 
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes.index(label)
        return img, label

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
])

test_dataset = MyDataset("/home/", "/home/chenhuil/hw/hw2/test_for_student.csv", transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
net = VGG().to(device)
net.load_state_dict(torch.load('/home/chenhuil/hw/hw2/VGG16_model_best.pth'))

# 测试模型
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('在测试集上的准确率： %d %%' % (100 * correct / total))
