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


class CreateModels:
    def __init__(self, n_model, epochs=2):
        self.model = n_model.to(device)
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=0.001)
        self.loss_func = nn.CrossEntropyLoss()

        self.epochs = epochs
        self.result = 0

    def train_res(self, train):
        self.model.train()
        for _e in range(self.epochs):
            loss_mean = 0
            lm_count = 0
            train_tqdm = tqdm(train, leave=False)
            for x_train, y_train in train_tqdm:
                x_train, y_train = x_train.to(device), y_train.to(device)
                predict = self.model(x_train)
                loss = self.loss_func(predict, y_train)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
                train_tqdm.set_description(f"Epoch [{_e + 1}/{self.epochs}], loss_mean={loss_mean:.3f}")

    def test_res(self, test):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            test_tqdm = tqdm(test, leave=False)
            for inputs, labels in test_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.result = 100 * correct / total

    def run(self):
        train, test = get_data()
        self.train_res(train)
        self.test_res(test)
        return self.result


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.F = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        self.outside = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_F = self.F(x)
        out_side = self.outside(x)
        return self.relu(out_F + out_side)


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    d_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    d_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_data = data.DataLoader(d_train, batch_size=4, shuffle=True)
    test_data = data.DataLoader(d_test, batch_size=4, shuffle=False)
    return train_data, test_data


def main():
    simple_cnn = CreateModels(SimpleCNN()).run()
    resnet = CreateModels(ResNet()).run()
    print(simple_cnn, resnet)


if __name__ == '__main__':
    main()
