from __future__ import print_function

import os
import argparse

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler,SGD
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import data
import utils
import metrics
import config_bayesian as cfg
from Models.BayesianModels.Bayesian3Conv3FC import BBB4Conv3FC
from Models.BayesianModels.BayesianResnet34 import BBBresnet34
from Models.BayesianModels.BayesianAlexNet import BBBAlexNet
from Models.BayesianModels.BayesianVGG11 import BBBVGG11


# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'resnet'):
        return BBBresnet34(priors, outputs, inputs, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '4conv3fc'):
        return BBB4Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'vgg11'):
        return BBB4Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [ResNet34 / AlexNet / 4conv3fc / vgg11')


def train_model(net, optimizer, criterion, trainloader, beta_type,num_ens=1):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    counter=0
    for i, (inputs, labels) in tqdm(enumerate(trainloader, 1)):
        counter+=1
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

       
        net_out, kl = net(inputs)
        outputs[:, :, 0] = F.log_softmax(net_out, dim=1)
        
        # kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(labels), beta_type)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs)*100, np.mean(kl_list)


def validate_model(net, criterion, validloader, beta_type,num_ens=1):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
       
        net_out, kl = net(inputs)

        outputs[:, :, 0] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(labels), beta_type)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))

    return valid_loss/len(validloader), np.mean(accs)*100


def run(dataset, net_type):

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    title= f'Bayes-{net_type}-{dataset}'
    writer = SummaryWriter(title)

    trainset, testset, inputs, outputs = data.getDataset(dataset,net_type)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'
    if os.path.exists(ckpt_name):
        net.load_state_dict(torch.load(ckpt_name))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    #criterion = metrics.ELBO(len(trainset)).to(device)
    criterion = metrics.ELBO(batch_size).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader,beta_type,num_ens=train_ens)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader,beta_type, num_ens=valid_ens)
        lr_sched.step(valid_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', train_loss, epoch)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='resnet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
