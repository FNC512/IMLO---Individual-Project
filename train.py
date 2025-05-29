import torch
import torchvision
import torch.nn.functional as f
import numpy as np
from torchvision.transforms import transforms
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import datetime as datetime

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.bn1(self.conv1(x))), (2, 2))
        x = self.do1(x)
        x = f.max_pool2d(f.relu(self.bn2(self.conv2(x))), 2)
        x = self.do1(x)
        x = f.max_pool2d(f.relu(self.bn3(self.conv3(x))), 2)
        x = self.do1(x)
        x = f.max_pool2d(f.relu(self.bn4(self.conv4(x))), 2)
        x = self.do1(x)
        x = f.max_pool2d(f.relu(self.bn5(self.conv5(x))), 2)
        x = self.do1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = self.do2(x)
        x = f.relu(self.fc2(x))
        x = self.do2(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_one_epoch(training_dl, model, loss_fn, optimizer):
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    running_loss = 0.
    total = 0
    correct = 0

    for i, data in enumerate(training_dl):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        total += labels.size(0)
        avg_loss = running_loss / len(training_dl)
        train_accuracy = correct / total
    return avg_loss, train_accuracy


#Adding a transform to normalize to data to optimal values
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

# Getting the CIFAR10  dataset
trainingdata = torchvision.datasets.CIFAR10(
    root='.data',
    train=True,
    download=True,
    transform=transform,
)

#Hyperparameters
learning_rate = 0.01
batch_size = 64
epochs = 110

#Get Validation and Training Data
size = len(trainingdata)
index = list(range(size))
np.random.shuffle(index)
split = int(np.floor(0.2 * size))
train_i, valid_i = index[split:], index[:split]

train_sampler = SubsetRandomSampler(train_i)
valid_sampler = SubsetRandomSampler(valid_i)
train_dl = torch.utils.data.DataLoader(trainingdata, batch_size=batch_size, sampler=train_sampler)
valid_dl = torch.utils.data.DataLoader(trainingdata, batch_size=batch_size, sampler=valid_sampler)

model = NeuralNetwork()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

epoch_number = 0

best_vloss = 1_000_000.

print(datetime.datetime.now())

for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    avg_loss, train_accuracy = train_one_epoch(train_dl, model, loss_fn, optimizer)
    running_vloss = 0.0
    vcorrect = 0
    vtotal = 0
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(valid_dl):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            vcorrect += (voutputs.argmax(1) == vlabels).type(torch.float).sum().item()
            vtotal += vlabels.size(0)

    avg_vloss = running_vloss / (i + 1)
    val_accuracy = vcorrect / vtotal
    print('LOSS train {:.4f} valid {:.4f} | TRAIN ACCURACY: {:2f}% | VAL ACCURACY: {:.2f}%'.format(avg_loss, avg_vloss, train_accuracy *100, val_accuracy * 100))

    scheduler.step(avg_vloss)

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), 'best_model.pth')
    epoch_number += 1

print(datetime.datetime.now())