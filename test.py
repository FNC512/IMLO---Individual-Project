import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as f
from train import NeuralNetwork


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

model = NeuralNetwork()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

loss_fn = nn.CrossEntropyLoss()

correct = 0
total = 0
running_loss = 0.0

with torch.no_grad():
    for i, data in enumerate(test_dl):
        inputs, labels = data
        outputs = NeuralNetwork.model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        total += labels.size(0)

avg_loss = running_loss / len(test_dl)
test_accuracy = correct / total

print('LOSS TEST {:.4f} | TEST ACCURACY: {:2f}%'.format(avg_loss, test_accuracy *100))

