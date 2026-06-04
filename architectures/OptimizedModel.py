import torch
import torch.nn as nn


class OptimizedModel(nn.Module):

    def __init__(
        self,
        conv_layers,
        normalizations,
        poolings,
        dropouts,
        initial_output_channel,
        initial_image_size,
        class_weights=None,
        dropout_p=0.3,
        num_classes=3
    ):
        super().__init__()

        # ---------------------------
        # validation
        # ---------------------------
        if len(normalizations) != conv_layers:
            raise ValueError("normalizations must match conv_layers")

        if len(poolings) != conv_layers:
            raise ValueError("poolings must match conv_layers")

        if len(dropouts) != conv_layers:
            raise ValueError("dropouts must match conv_layers")

        self.image_size = initial_image_size
        self.optimizer = None  # will be set later

        # ---------------------------
        # CNN builder
        # ---------------------------
        layers = []

        in_channels = 3
        out_channels = initial_output_channel

        for i in range(conv_layers):

            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ))

            if normalizations[i]:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(nn.ReLU())

            if poolings[i]:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.image_size //= 2

            if dropouts[i]:
                layers.append(nn.Dropout2d(p=dropout_p))

            in_channels = out_channels
            out_channels *= 2

        self.conv = nn.Sequential(*layers)

        # ---------------------------
        # classifier
        # ---------------------------
        flattened_size = in_channels * self.image_size * self.image_size
        self.fc = nn.Linear(flattened_size, num_classes)

        # ---------------------------
        # loss
        # ---------------------------
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # forward
    # ---------------------------
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # ---------------------------
    # loss
    # ---------------------------
    def calculate_loss(self, y_pred, y):
        return self.criterion(y_pred, y)

    # ---------------------------
    # SETTER METHOD (NEW)
    # ---------------------------
    def set_optimizer(self, optimizer):
        """
        Attach optimizer externally (supports Adam, SGD, OneCycleLR setups).
        """
        self.optimizer = optimizer

    # ---------------------------
    # backward (your old API kept)
    # ---------------------------
    def backward(self, loss, lr=None):
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()