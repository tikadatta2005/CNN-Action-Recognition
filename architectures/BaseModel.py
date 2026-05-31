import torch
import torch.nn as nn

class BaseModel (nn.Module):
    def __init__(
        self,
        conv_layers,
        initial_output_channel,
        initial_image_size,
        ):
        
        #initialize nn.Module
        super().__init__()
        
        # set conv layers for future use
        self.conv_layers = conv_layers
        
        # default parameters
        # Conv2D parameters
        in_channels = 3 #RGB
        out_channels = initial_output_channel
        
        #image size for activation
        final_image_size = initial_image_size
                    
        # make conv layers
        for i in len(conv_layers):
            
            # conv layer
            self["K"+str(i)] = nn.Conv2d(
                in_channels=in_channel,
                out_channels= out_channels,
                stride=1,
                kernel_size= 3,
                padding=1                
            )
            
            # pool layer
            self["pool"+str(i)] = nn.MaxPool2d(2, 2)            
            
            final_image_size = final_image_size//2
            in_channels = out_channels
            out_channels = out_channels * 2
        
        # Linear Layer
        self.W = nn.Parameter(torch.randn((in_channels * final_image_size * final_image_size, 3)) * 0.01)
        self.b = nn.Parameter(torch.zeros(3))
        
        #Loss function
        criterion = nn.CrossEntropyLoss()
        
        # forward propagation
        def forward(self, x):
            
            # move forward
            for i in len(self.conv_layers):
                x = self["K"+str(i)](x)
                x = nn.functional.relu(x)
                x = self["pool"+str(i)](x)
            
            # call linear layer
            x = x @ self.W + self.b
            
            # return x
            return x  
        
        # calculate loss
        def calculate_loss(self, y_hat, y):
            loss = self.criterion(y_hat, y)
            return loss
        
        # calculate backward propagation
        def backward(self, loss, lr = 0.001):
            
            # track back loss
            loss.backward()
            
            # use torch.no_grad for stop updating grad during back propagation
            with torch.no_grad():
                # update the gradients
                self.W -= lr * self.W.grad
                self.W.grad.zero_()
                
                self.b -= lr * self.b.grad
                self.b.grad.zero_()
            
            return loss.item()