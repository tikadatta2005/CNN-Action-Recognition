import torch
import torch.nn as nn
import torch.nn.functional as F


class NormModel(nn.Module):

    def __init__(self, conv_layers, initial_output_channel, initial_image_size):

        # initialize nn.Module
        super().__init__()

        # store image size for flatten calculation
        self.image_size = initial_image_size

        # default channels
        in_channels = 3  # RGB
        out_channels = initial_output_channel

        # store all layers here
        layers = []

        # build conv layers dynamically
        for _ in range(conv_layers):

            # conv layer
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            
            # batch norm layer
            layers.append(nn.BatchNorm2d(out_channels))

            # activation
            layers.append(nn.ReLU())
            

            # downsample image
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            #dropout layer
            layers.append(nn.Dropout2d(p=0.3))

            # update channels for next layer
            in_channels = out_channels
            out_channels = out_channels * 2

            # reduce image size after pooling
            self.image_size = self.image_size // 2

        # combine conv layers
        self.conv = nn.Sequential(*layers)

        # flatten size calculation
        flattened_size = in_channels * self.image_size * self.image_size

        # final linear layer for classification
        self.W = nn.Parameter(torch.randn(flattened_size, 3) * 0.01)
        self.b = nn.Parameter(torch.zeros(3))

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):

        # pass through conv layers
        x = self.conv(x)

        # flatten for linear layer
        x = x.view(x.size(0), -1)

        # final classification layer
        x = x @ self.W + self.b

        return x

    def calculate_loss(self, y_pred, y):

        # compute cross entropy loss
        loss = self.criterion(y_pred, y)

        return loss
    
    def backward(self, loss, lr=0.001):
        
        # track backward
        loss.backward()
        
        # with no grad
        with torch.no_grad():
            
            for param in self.parameters():

                if param.grad is not None:
                    param -= lr * param.grad
                    param.grad.zero_()
            
        return loss.item()