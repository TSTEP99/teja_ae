"""Implementation for the decoder portion of Teja-VAE"""
from helper import reparameterization
from math import floor
import torch
import torch.nn as nn

class teja_decoder(nn.Module):
    def __init__(self, other_dims,  hidden_layer_size = 100, rank = 3, device = None):
        """Initializes the parameters and layers for the neural network for Teja-VAE"""

        #Calls constructor of super class
        super(teja_decoder, self).__init__()

        #defines the device for the module
        if device:
            self.device = device
        else:
            self.device = "cpu"

        #Initializes other factor matrices used in the encoder portion
        other_factors=[]

        for dim in other_dims:
            other_factors.append(nn.Parameter(torch.randn((dim, rank), requires_grad=True, device = device))) 

        self.other_factors = nn.ParameterList(other_factors)

        # Layers for computing the individual tensor elements
        self.FC_input = nn.Linear((len(other_dims)+1) * rank, hidden_layer_size, device = device)
        self.FC_element = nn.Linear(hidden_layer_size, 1, device = device)

        #Activation function to compute the hidden layer when decoding
        self.tanh = nn.Tanh()    
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()

        #Saves the size(s) of the non-epoch/non-sample dimensions
        self._other_dims = other_dims
    
    def forward(self, latent_space):
        """Computes the forward pass for Teja-VAE takes sample of Teja-VAE encoder as input"""

        dims = []
        dims.append(latent_space.shape[0])
        dims.extend(self._other_dims)

        indices = self._create_indices(dims)

        num_dims = indices.shape[1]

        #Makes array(s) with all the factor means and log variances
        factors = []

        factors.append(latent_space)

        factors.extend(self.other_factors)

        #creates tensor to form u vector(s) from VAE-CP paper
        Us = []

        #Samples to create aforementioned u vector
        for i in range(num_dims):
            Us.append(factors[i][indices[:,i]])

        #Concatenates tensors to form u vector from paper 
        U_vecs = torch.concat(Us, dim=1)

        #Pass u vector through the decoder layers to generate mean and log_var value for each tensor element
        #Note: The original paper uses the tanh function for the hidden layer
        hidden = self.relu(self.FC_input(U_vecs))


        #NOTE: use ReLU Activation
        elements = self.sigmoid(self.FC_element(hidden))

        # #Samples on a per element basis using output of decoder layers
        # sample_elements = self._reparameterization(elements_mean, elements_log_var)

        #Reshape mean and log_var to original tensor size

        return_tensor = elements.view(*dims)

        return return_tensor
        

    def _create_indices(self,dims):
        """
        Takes tensor shape as input and 
        creates list of all possible indices
        """
        
        indices = []

        for dim in dims:
            indices.append(torch.arange(dim))
        indices = torch.cartesian_prod(*indices)
        return indices.long() 
