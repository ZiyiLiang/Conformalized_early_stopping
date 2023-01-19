from turtle import forward
from xml.sax.xmlreader import InputSource
import numpy as np
import torch as th 
import torch.nn as nn
import torch.nn.functional as F
import pdb


class SimpleConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(SimpleConvolutionalNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(18 * 16 * 16, 64) 
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Forward pass,
        x shape is (batch_size, 3, 32, 32)
        (color channel first)
        in the comments, we omit the batch_size in the shape
        """
        # shape : 3x32x32 -> 18x32x32
        x = F.relu(self.conv1(x))
        # 18x32x32 -> 18x16x16
        x = self.pool(x)
        # 18x16x16 -> 4608
        x = x.view(-1, 18 * 16 * 16)
        # 4608 -> 64
        x = F.relu(self.fc1(x))
        # 64 -> 10
        # The softmax non-linearity is applied later (cf createLossAndOptimizer() fn)
        x = self.fc2(x)
        return x
    
    def predict_prob(self, inputs):
        """
        Predict probabilities given any input data
        """
        self.eval()
        if len(inputs.shape) ==3:
            inputs = inputs[None]
            
        get_prob = nn.Softmax(dim = 1)
        with th.no_grad():
            logits = self(inputs)    # Assume inputs are already on the correct device
            prob = get_prob(logits).cpu().numpy()
        return prob



class ConvAutoencoder_32(nn.Module):
    def __init__(self):
        super().__init__()
        ## encoder layers ##
        # channels: 1 -> 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        # channels: 16 -> 4
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer with kernel of 2 and stride of 2 will decrease the dim by a factor of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by a factor of 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        # shape: 3x32x32 -> 16x32x32
        x = F.relu(self.conv1(x))
        # 16x32x32 -> 16x16x16
        x = self.pool(x)
        # add second hidden layer
        # 16x16x16 -> 4x16x16
        x = F.relu(self.conv2(x))
        # 4x16x16 -> 4x8x8
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        # shape: 4x8x8 -> 16x16x16
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        # 16x16x16 -> 3x32x32
        x = th.sigmoid(self.t_conv2(x))
        return x

            
    def get_anomaly_scores(self, inputs):
        """
        Compute the anomaly scores for a given set of inputs as the rescontruction error
        """

        self.eval()
        with th.no_grad():
            outputs = self(inputs)
            Loss = th.nn.MSELoss(reduction='none')
            scores = Loss(outputs.reshape(-1, 3 * 32 * 32), inputs.reshape(-1, 3 * 32 * 32))
        
        scores = np.mean(scores.numpy(), axis = 1)
        return list(scores)
 


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # channels: 1 -> 16
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        # channels: 16 -> 4
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer with kernel of 2 and stride of 2 will decrease the dim by a factor of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by a factor of 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        # shape: 1x28x28 -> 16x28x28
        x = F.relu(self.conv1(x))
        # 16x28x28 -> 16x14x14
        x = self.pool(x)
        # add second hidden layer
        # 16x14x14 -> 4x14x14
        x = F.relu(self.conv2(x))
        # 4x14x14 -> 4x7x7
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        # shape: 4x7x7 -> 16x14x14
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        # 16x14x14 -> 1x28x28
        x = th.sigmoid(self.t_conv2(x))
        return x

            
    def get_anomaly_scores(self, inputs):
        """
        Compute the anomaly scores for a given set of inputs as the rescontruction error
        """
        if len(inputs.shape) ==3:
            inputs = inputs[None]

        self.eval()
        with th.no_grad():
            outputs = self(inputs)
            Loss = th.nn.MSELoss(reduction='none')
            scores = Loss(outputs.reshape(-1, 28 * 28), inputs.reshape(-1, 28 * 28))
        
        scores = np.mean(scores.numpy(), axis = 1)
        return list(scores)
    
    
class mse_model(nn.Module):
    """ Deep conditional mean regression minimizing MSE loss
    """

    def __init__(self,
                 in_shape=1,
                 hidden_size=64,
                 dropout=0.5):
        """ Initialization

        Parameters
        ----------

        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """

        super().__init__()
        self.in_shape = in_shape
        self.out_shape = 1
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        """ Construct the network
        """
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            # nn.Dropout(self.dropout),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(),
            # nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        return th.squeeze(self.base_model(x))

def MSE_loss(outputs, inputs, targets):
  return th.mean((outputs - targets)**2)
