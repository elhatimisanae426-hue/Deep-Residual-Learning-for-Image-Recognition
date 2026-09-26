# ResNet Implementation on CIFAR-10
# Implementation of Deep Residual Learning (He et al., 2015) for CIFAR-10 classification
# Paper: arXiv:1512.03385v1

# === 1. Install Dependencies ===
# !pip install torch torchvision matplotlib

# === 2. Import Libraries ===
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# === 3. ResNet Building Block (BasicBlock) ===
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = torch.relu(out)
        return out

# === 4. ResNet Model ===
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# === 5. Create Model Variants ===
def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])

def ResNet32():
    return ResNet(BasicBlock, [5, 5, 5])

def ResNet44():
    return ResNet(BasicBlock, [7, 7, 7])

def ResNet56():
    return ResNet(BasicBlock, [9, 9, 9])

def ResNet110():
    return ResNet(BasicBlock, [18, 18, 18])

# === 6. Data Loading and Preprocessing ===
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# === 7. Training Function ===
def train(model, device, trainloader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(trainloader)} | Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
    return running_loss/len(trainloader), 100.*correct/total

# === 8. Testing Function ===
def test(model, device, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print(f'Test Loss: {test_loss/len(testloader):.4f} | Test Acc: {acc:.2f}%')
    return test_loss/len(testloader), acc

# === 9. Train ResNet-20 on CIFAR-10 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = ResNet20().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1)

num_epochs = 160
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, device, trainloader, optimizer, criterion, epoch)
    test_loss, test_acc = test(model, device, testloader, criterion)
    scheduler.step()
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# === 10. Compare Different Depths (Table 6 from paper) ===
models_info = {
    'ResNet-20': {'layers': [3,3,3], 'params_M': 0.27, 'error': '8.75%'},
    'ResNet-32': {'layers': [5,5,5], 'params_M': 0.46, 'error': '7.51%'},
    'ResNet-44': {'layers': [7,7,7], 'params_M': 0.66, 'error': '7.17%'},
    'ResNet-56': {'layers': [9,9,9], 'params_M': 0.85, 'error': '6.97%'},
    'ResNet-110': {'layers': [18,18,18], 'params_M': 1.7, 'error': '6.43%'},
    'ResNet-1202': {'layers': [200,200,200], 'params_M': 19.4, 'error': '7.93%'}
}

for name, info in models_info.items():
    print(f"{name}: {info['params_M']}M params, CIFAR-10 Error: {info['error']}")

# === Results from the paper ===
# ResNet-20: 0.27M params, 8.75% error
# ResNet-32: 0.46M params, 7.51% error
# ResNet-44: 0.66M params, 7.17% error
# ResNet-56: 0.85M params, 6.97% error
# ResNet-110: 1.7M params, 6.43% error
# ResNet-1202: 19.4M params, 7.93% error
