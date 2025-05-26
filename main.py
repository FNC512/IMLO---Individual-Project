import torch
import torchvision
from torchvision.transforms import ToTensor
from torch import nn
from torch.utils.data import DataLoader
from NN import NeuralNetwork
from training import train, test

# Getting the CIFAR10  datasets
trainingdata = torchvision.datasets.CIFAR10(
    root='.data',
    train=True,
    download=True,
    transform=ToTensor(),
)

testdata = torchvision.datasets.CIFAR10(
    root='.data',
    train=False,
    download=True,
    transform=ToTensor(),
)

#Hyperparameters
learning_rate = 1e-2
batch_size = 64
epochs = 20

# Create data loaders.
train_dl = DataLoader(trainingdata, batch_size=batch_size)
test_dl = DataLoader(testdata, batch_size=batch_size)
classes = {0:"plane", 1:"car", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

model = NeuralNetwork()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)



for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer, batch_size)
    test_loss = test(test_dl, model, loss_fn)
    scheduler.step(test_loss)
print("Done!")




test(test_dl, model, loss_fn)

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth", weights_only=True))