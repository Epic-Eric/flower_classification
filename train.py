import sys
import argparse
from model import build_model
import image_loader
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data.dataloader

def train(args: argparse.Namespace):
    # Load data
    trainloader, testloader, validationloader, class_to_idx = image_loader.image_loader(data_directory)
    model = build_model(args)

    device = torch.device("mps" if torch.backends.mps.is_built() and args.gpu else "cpu")
    model.to(device)
    print("Device: ", device)

    # Start training
    epoch = args.epochs
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    for i in range(epoch):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
        total_loss = 0
        total_accuracy = 0
        model.eval()
        for images, labels in testloader:
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device)
                logps = model.forward(images)
                loss = criterion(logps, labels)
                total_loss += loss.item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equal = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equal.type(torch.FloatTensor))
                total_accuracy += accuracy
        model.train()
        
        print(f"Epoch: {i+1}/{epoch}")
        print(f"Test loss: {total_loss / len(testloader) :.3f}")
        print(f"Test accuracy: {total_accuracy / len(testloader) * 100 :.3f}%")

    # Model final accuracy
    model.eval()
    with torch.no_grad():
        total_accuracy = 0
        for images, labels in validationloader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equal = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equal.type(torch.FloatTensor))
            total_accuracy += accuracy
    print(f"Model's final accuracy: {total_accuracy / len(validationloader) * 100 :.3f}%")
    # Accuracy = 86.510%

    # Save the model
    model.class_to_idx = class_to_idx
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'class_to_idx': model.class_to_idx,
        'arch': args.arch,
        'hidden_units': args.hidden_units
    }
    path = f"{args.save_dir}/{args.arch}_checkpoint.pth"
    torch.save(checkpoint, path)


if __name__ == "__main__":
    # Argument parsing
    data_directory = ""
    if (len(sys.argv) > 1):
        data_directory = sys.argv[1]
    else:
        print("Please provide data directory as argument")
        sys.exit(1)
    sys.argv = sys.argv[:1] + sys.argv[2:]

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--save_dir', type=str, default="checkpoints")
    argparser.add_argument('--arch', type=str, default="vgg19")
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--hidden_units', type=int, default=1024)
    argparser.add_argument('--epochs', type=int, default=10)
    argparser.add_argument('--gpu', action='store_true')
    args = argparser.parse_args()

    train(args)