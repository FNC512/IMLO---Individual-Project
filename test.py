import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as f

# Establishing NeuralNetwork Achitecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Adding Covolutional Layers with corresponding batch normalisation
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
        # Adding fully connected layers and dropout
        self.fc1 = nn.Linear(512 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        # Applying batch normalisation, activation function and max pooling on the output of each convolutional layer
        x = f.max_pool2d(f.relu(self.bn1(self.conv1(x))), (2, 2))
        # Applying dropout to each output
        x = self.do1(x)
        x = f.max_pool2d(f.relu(self.bn2(self.conv2(x))), 2)
        x = self.do1(x)
        x = f.max_pool2d(f.relu(self.bn3(self.conv3(x))), 2)
        x = self.do1(x)
        x = f.max_pool2d(f.relu(self.bn4(self.conv4(x))), 2)
        x = self.do1(x)
        x = f.max_pool2d(f.relu(self.bn5(self.conv5(x))), 2)
        x = self.do1(x)

        # Flattening the data then passing it through each fully connected layer, with dropout layers between
        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = self.do2(x)
        x = f.relu(self.fc2(x))
        x = self.do2(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # Counting the number of features post flattening
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#Adding a transform to normalize to data to optimal values
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])


batch_size = 64

# Getting the CIFAR10  dataset
testdata = torchvision.datasets.CIFAR10(
    root='.data',
    train=False,
    download=True,
    transform=transform,
)

#test dataloader
test_dl = DataLoader(testdata, batch_size=batch_size)

#Loading the best model
model = NeuralNetwork()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

loss_fn = nn.CrossEntropyLoss()
correct = 0
total = 0
running_loss = 0.0

#Evaluating
with torch.no_grad():
    for i, data in enumerate(test_dl):
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        total += labels.size(0)

avg_loss = running_loss / len(test_dl)
test_accuracy = correct / total

print('LOSS TEST {:.4f} | TEST ACCURACY: {:2f}%'.format(avg_loss, test_accuracy *100))

