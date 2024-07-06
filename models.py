# coding=utf-8
"""Models."""

import torch
from torch import nn

class Net(nn.Module):
    def __init__(self,dropout_prob=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 256 * 56 * 56)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_prob)
        x = self.fc2(x)
        return x

class MyModel(nn.Module):
  def __init__(self, image_size, num_class, image_channel=3, drop_out_rate=0.3):
    super().__init__()

    self.image_size = image_size
    self.num_class = num_class
    self.image_channel = image_channel

    layers = [
        nn.Conv2d(
            in_channels=image_channel, out_channels=64,
            kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64, momentum=0.1),
        nn.ReLU(),
        nn.Dropout(drop_out_rate),

        nn.Conv2d(64, 64, 3, 1, 1),
        nn.BatchNorm2d(64, momentum=0.1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, 1, 1),
        nn.BatchNorm2d(64, momentum=0.1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, 3, 1, 1),
        nn.BatchNorm2d(128, momentum=0.1),
        nn.ReLU(),
        nn.Dropout(drop_out_rate),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.BatchNorm2d(128, momentum=0.1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(128, 256, 3, 1, 1),
        nn.BatchNorm2d(256, momentum=0.1),
        nn.ReLU(),
        nn.Dropout(drop_out_rate),

        # layer 7
        nn.Conv2d(256, 256, 3, 1, 1),
        nn.BatchNorm2d(256, momentum=0.1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(256, 256, 3, 1, 1),
        nn.BatchNorm2d(256, momentum=0.1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # layer 9
        nn.Conv2d(256, 256, 3, 1, 1),
        nn.BatchNorm2d(256, momentum=0.1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 64x downsampled now

        nn.Flatten(),  # simple squeeze
    ]

    self.layers = nn.Sequential(*layers)
    self.emb_dim = 256
    self.final_layer = nn.Linear(self.emb_dim, num_class)

  def forward(self, x):
    feature = self.layers(x)
    class_pred = self.final_layer(feature)
    return feature, class_pred


# with skip connections
class MyModel2(nn.Module):
  def __init__(self, image_size, num_class,
               image_channel=3, use_skip=False,
               drop_out_rate=0.3,
               channel_base=64):
    super().__init__()

    self.image_size = image_size
    self.num_class = num_class
    self.image_channel = image_channel
    self.use_skip = use_skip

    c_base = channel_base
    drop_out_rate = drop_out_rate

    def make_layers(in_channels, out_channels, drop_out=0.0, maxpool=0):
      layers = [
          nn.Conv2d(
              in_channels=in_channels, out_channels=out_channels,
              kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(out_channels, momentum=0.1),
          nn.ReLU(),
      ]
      if drop_out > 0.0:
        layers += [nn.Dropout(drop_out_rate)]
      if maxpool:
        layers += [nn.MaxPool2d(kernel_size=2, stride=2) for i in range(maxpool)]
      return nn.Sequential(*layers)

    self.layer1 = make_layers(
        image_channel, c_base, drop_out=drop_out_rate, maxpool=0)
    self.layer2 = make_layers(
        c_base, c_base, drop_out=0.0, maxpool=1)
    self.layer3 = make_layers(
        c_base, c_base, drop_out=0.0, maxpool=1)

    self.layer4 = make_layers(
        c_base, c_base*2, drop_out=drop_out_rate, maxpool=0)
    self.layer34_trans = make_layers(c_base, c_base*2, maxpool=1)

    self.layer5 = make_layers(
        c_base*2, c_base*2, drop_out=0.0, maxpool=1)

    self.layer6 = make_layers(
        c_base*2, c_base*4, drop_out=drop_out_rate, maxpool=0)
    self.layer56_trans = make_layers(c_base*2, c_base*4, maxpool=1)

    self.layer7 = make_layers(
        c_base*4, c_base*4, drop_out=0.0, maxpool=1)
    self.layer8 = make_layers(
        c_base*4, c_base*4, drop_out=0.0, maxpool=1)

    self.layer9 = make_layers(
        c_base*4, c_base*8, drop_out=drop_out_rate, maxpool=0)

    self.layer10 = make_layers(
        c_base*8, c_base*8, drop_out=0.0, maxpool=1)
    self.layer78910_trans = make_layers(c_base*4, c_base*8, maxpool=3)

    self.flatten = nn.Flatten()
    self.emb_dim = c_base*8
    self.final_layer = nn.Linear(self.emb_dim, num_class)

  def forward(self, x):
    use_skip = self.use_skip

    if use_skip:
      feature = self.layer1(x)
      feature = self.layer2(feature)

      feature3 = feature
      feature = self.layer3(feature)
      feature = self.layer4(feature) + self.layer34_trans(feature3)

      feature5 = feature
      feature = self.layer5(feature)
      feature = self.layer6(feature) + self.layer56_trans(feature5)

      feature7 = feature
      feature = self.layer7(feature)
      feature = self.layer8(feature)
      feature = self.layer9(feature)
      feature = self.layer10(feature) + self.layer78910_trans(feature7)

    else:
      feature = self.layer1(x)
      feature = self.layer2(feature)
      feature = self.layer3(feature)

      feature = self.layer4(feature)
      feature = self.layer5(feature)
      feature = self.layer6(feature)
      #if use_skip:
      #  feature += self.block1_trans(block1_feat)

      feature = self.layer7(feature)
      feature = self.layer8(feature)
      feature = self.layer9(feature)
      feature = self.layer10(feature)

    feature = self.flatten(feature)
    class_pred = self.final_layer(feature)
    return feature, class_pred

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.pool3(x)
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.conv10(x)))
        x = self.pool4(x)
        x = self.relu(self.bn11(self.conv11(x)))
        x = self.relu(self.bn12(self.conv12(x)))
        x = self.relu(self.bn13(self.conv13(x)))
        x = self.pool5(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class VGG_skip(nn.Module):
    def make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
            in_channels = out_channels
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def __init__(self, num_classes=1000):
        super(VGG_skip, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 256, 3)
        self.layer4 = self.make_layer(256, 512, 3)
        self.layer5 = self.make_layer(512, 512, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.bn6 = nn.BatchNorm1d(4096)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        res1 = out
        out = self.layer1(out)
        out += res1
        out = nn.ReLU(inplace=True)(out)
        
        res2 = out
        out = self.layer2(out)
        out += res2
        out = nn.ReLU(inplace=True)(out)
        
        res3 = out
        out = self.layer3(out)
        out += res3
        out = nn.ReLU(inplace=True)(out)
        
        res4 = out
        out = self.layer4(out)
        out += res4
        out = nn.ReLU(inplace=True)(out)
        
        res5 = out
        out = self.layer5(out)
        out += res5
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.bn6(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.bn7(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.drop2(out)

        out = self.fc3(out)

        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.5)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out


def ResNet18():
    return ResNet(ResBlock, [2, 2, 2, 2])
