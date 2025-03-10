import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import ssl
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    d_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    d_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)

    train_data = data.DataLoader(d_train, batch_size=4, shuffle=True)
    test_data = data.DataLoader(d_test, batch_size=4, shuffle=False)
    return train_data, test_data


def model_simplecnn(train, test):
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(params=model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    model = train_res(loss_func, train, model, optimizer)

    return test_res(model, test, "SimpleCNN")


def resnet18(test):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)

    return test_res(model, test, "ResNet 18")


def resnet34(test):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)

    return test_res(model, test, "ResNet 34")


def resnet152(test):
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)

    return test_res(model, test, "ResNet 152")


def train_res(loss_func, train, model, optimizer, epochs=2):
    model.train()

    for _e in range(epochs):
        loss_mean = 0
        lm_count = 0
        train_tqdm = tqdm(train, leave=False)
        for x_train, y_train in train_tqdm:
            predict = model(x_train)
            loss = loss_func(predict, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
            train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")
    return model


def test_res(model, test, name):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        test_tqdm = tqdm(test, leave=False)
        for inputs, labels in test_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model {name}: {100 * correct / total:.2f}%')


def main():
    train, test = get_data()
    model_simplecnn(train, test)
    resnet18(test)
    resnet34(test)
    resnet152(test)


if __name__ == '__main__':
    main()
