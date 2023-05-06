import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def getDataset(dataset,model):
    # if(model=='vgg'):
    #     transform_mnist = transforms.Compose([
    #         transforms.Resize((64, 64)),
    #         transforms.ToTensor(),
    #         ])
    # elif(model=='cnn'):
    #     transform_mnist = transforms.Compose([
    #         transforms.Resize((64, 64)),
    #         transforms.ToTensor(),
    #         ])
    #     transform_cifar = transforms.Compose([
    #         transforms.Resize((64, 64)),
    #         transforms.ToTensor(),
    #         ])
    # elif(model=='nalexnet'):
    #     transform_cifar = transforms.Compose([transforms.Resize((70, 70)),
    #                                    transforms.RandomCrop((64, 64)),
    #                                    transforms.ToTensor()])
    #     transform_mnist = transforms.Compose([transforms.Resize((70, 70)),
    #                                    transforms.RandomCrop((64, 64)),
    #                                    transforms.ToTensor()])
    # else:
    transform_mnist = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    transform_cifar = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    if(dataset == 'CIFAR10'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
        num_classes = 10
        inputs=3

    elif(dataset == 'CIFAR100'):
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_cifar)
        num_classes = 100
        inputs = 3
        
    elif(dataset == 'MNIST'):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
        num_classes = 10
        inputs = 1
        
    return trainset, testset, inputs, num_classes


def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader, valid_loader, test_loader
