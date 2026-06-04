import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicModel(nn.Module):

    def __init__(
        self,
        conv_layers,
        normalizations,
        poolings,
        dropouts,
        initial_output_channel,
        initial_image_size,
        activations,
        activation_outputs,
        dropout_p=0
    ):

        # initialize nn.Module
        super().__init__()

        if not normalizations or len(normalizations) != conv_layers:
            print("Normalization accepts array with equal length of conv_layers!")

        if not poolings or len(poolings) != conv_layers:
            print("Poolings accepts array with equal length of conv_layers!")

        if not dropouts or len(dropouts) != conv_layers:
            print("Dropouts accepts array with equal length of conv_layers!")

        # store image size for flatten calculation
        self.image_size = initial_image_size

        # default channels
        in_channels = 3
        out_channels = initial_output_channel

        # store conv layers here
        conv_block = []

        # build conv layers dynamically
        for i in range(conv_layers):

            # conv layer
            conv_block.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )

            # batch norm
            if normalizations[i] == 1 or normalizations[i] is True:
                conv_block.append(nn.BatchNorm2d(out_channels))

            # activation
            conv_block.append(nn.ReLU())

            # pooling
            if poolings[i] == 1 or poolings[i] is True:
                conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.image_size = self.image_size // 2

            # dropout
            if dropouts[i] == 1 or dropouts[i] is True:
                conv_block.append(nn.Dropout2d(p=dropout_p))

            # update channels
            in_channels = out_channels
            out_channels = out_channels * 2

        # conv feature extractor
        self.conv = nn.Sequential(*conv_block)

        # flatten size after conv layers
        flattened_size = (
            in_channels *
            self.image_size *
            self.image_size
        )

        allowed = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }

        # dense layers
        fc_layers = []

        current_features = flattened_size

        if activations and len(activations) > 0:

            for activation, output_val in zip(activations, activation_outputs):

                if activation not in allowed:
                    continue

                fc_layers.append(
                    nn.Linear(
                        current_features,
                        output_val
                    )
                )

                fc_layers.append(
                    allowed[activation]
                )

                current_features = output_val

        self.fc = nn.Sequential(*fc_layers)

        # final classification layer
        self.W = nn.Parameter(
            torch.randn(current_features, 3) * 0.01
        )

        self.b = nn.Parameter(
            torch.zeros(3)
        )

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):

        # conv layers
        x = self.conv(x)

        # flatten
        x = x.view(x.size(0), -1)

        # dense layers
        x = self.fc(x)

        # final classification
        x = x @ self.W + self.b

        return x

    def calculate_loss(self, y_pred, y):

        # compute cross entropy loss
        loss = self.criterion(y_pred, y)

        return loss

    def backward(self, loss, lr=0.001):

        # backward pass
        loss.backward()

        with torch.no_grad():

            for param in self.parameters():

                if param.grad is not None:

                    param -= lr * param.grad
                    param.grad.zero_()

        return loss.item()