# data_preparation.py (modifié)
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
