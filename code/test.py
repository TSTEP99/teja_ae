"""
File to test the functionality of all implemented modules in Teja-VAE
Note: These tests are now invalid will need to update in the future
"""
from teja_encoder import teja_encoder
from teja_decoder import teja_decoder
from teja_ae import teja_ae
import torch

def test_encoder():
    """Function to test the teja encoder module"""

    sample_tensor = torch.randn(14052, 19, 45)
    other_dims = sample_tensor.shape[1:]
    encoder = teja_encoder(other_dims = other_dims)
    latent_space = encoder(sample_tensor)

    assert tuple(latent_space.shape) == (14052, 3)

def test_decoder():
    """Function to test the teja decoder module"""

    latent_space = torch.randn((14052,3))
    other_dims = [19, 45]

    decoder = teja_decoder(other_dims = other_dims)
    return_tensor =decoder(latent_space)

    assert tuple(return_tensor.shape) == (14052, 19, 45)

def test_vae():
    """Function to test Teja-VAE"""

    device = "cuda:0"
    sample_tensor = torch.randn(14052, 19, 45, device = device)
    other_dims = sample_tensor.shape[1:]
    ae = teja_ae(other_dims = other_dims, device = device)
    return_tensor = ae(sample_tensor)

    assert sample_tensor.shape == return_tensor.shape

if __name__ == "__main__":
    
    test_encoder()
    test_decoder()
    test_vae()
