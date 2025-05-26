import torch
import torchvision
import torch.nn.functional as f
from torchvision.transforms import ToTensor
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def train(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def train_one_epoch(epoch_index, tb_writer, training_dl, model, loss_fn, optimizer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_dl):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(training_dl)


# Getting the CIFAR10  dataset
trainingdata = torchvision.datasets.CIFAR10(
    root='.data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# Getting the CIFAR10  dataset
testdata = torchvision.datasets.CIFAR10(
    root='.data',
    train=False,
    download=True,
    transform=ToTensor(),
)


#Hyperparameters
learning_rate = 1e-2
batch_size = 64
epochs = 30

# Create data loaders.
train_dl = DataLoader(trainingdata, batch_size=batch_size)
test_dl = DataLoader(testdata, batch_size=batch_size)

model = NeuralNetwork()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

epoch_number = 0

best_vloss = 1_000_000.

for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer, train_dl, model, loss_fn, optimizer)
    running_vloss = 0.0
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(test_dl):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            correct += (voutputs.argmax(1) == vlabels).type(torch.float).sum().item()
            total += vlabels.size(0)

    avg_vloss = running_vloss / (i + 1)
    accuracy = correct / total
    print('LOSS train {:.4f} valid {:.4f} | VAL ACCURACY: {:.2f}%'.format(avg_loss, avg_vloss, accuracy * 100))

    scheduler.step(avg_vloss)

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), 'best_model.pth')

    epoch_number += 1

#torch.save(model.state_dict(), "model.pth")
#print("Saved PyTorch Model State to model.pth")
#model = NeuralNetwork()
#model.load_state_dict(torch.load("model.pth", weights_only=True))