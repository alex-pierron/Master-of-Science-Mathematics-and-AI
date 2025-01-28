## Standard libraries
import os
import math
import numpy as np 
import time
from PIL import Image
import shutil
import pandas as pd

## Imports for plotting
from matplotlib import use
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()
## Progress bar
from tqdm import tqdm

## PyTorch
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


class MLP(nn.Module):
    """
        Implémente un réseau de neurones à couches multiples (MLP).

        Paramètres :
        - input_size (int) : Dimensionnalité de l'entrée du réseau.
        - hidden_sizes (list) : Liste des tailles des couches cachées.
        - activation_function (torch.nn.Module) : Fonction d'activation à utiliser entre les couches cachées.

        Attributs :
        - layers (torch.nn.ModuleList) : Liste des couches cachées.
        - activations (torch.nn.ModuleList) : Liste des fonctions d'activation entre les couches cachées.
        - activation (torch.nn.Module) : Fonction d'activation à utiliser.

        Méthode forward :
        - forward(x) : Propagation avant à travers le réseau.

    """
    
    def __init__(self, input_size, hidden_sizes, activation_function=nn.Sigmoid()):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.activation = activation_function

        # Ajoute les couches cachées
        in_features = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
            self.activations.append(activation_function)

        # Ajoute la couche de sortie
        self.output_layer = nn.Linear(in_features, 1)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))

        x = self.output_layer(x)
        return x

def initialize_weights(model, mean, std, seed=None):
    """
    Initialise les poids du modèle selon une distribution normale avec la moyenne et l'écart type spécifiés.

    Paramètres :
    - model (torch.nn.Module) : Le modèle à initialiser.
    - mean (float) : Moyenne de la distribution normale.
    - std (float) : Écart type de la distribution normale.
    - seed (int) : Graine pour la génération de nombres aléatoires.

    Renvoie :
    - model (torch.nn.Module) : Le modèle avec les poids initialisés.

    """
    
    if seed is not None:
        torch.manual_seed(seed)

    for name, param in model.named_parameters():
        if 'layers' or 'output_layer' in name:  # Couches cachées
            if 'weight' in name:
                init.normal_(param, mean=mean, std=std)
            elif 'bias' in name:
                init.constant_(param, 0)
        elif  'input_layer' in name:  # Couches d'entrée et de sortie
            if 'weight' in name:
                init.xavier_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0)
    return model


def calculate_l1_norm(model,checkpoint_path, print_norm=True):
    """
    Calcule la norme L1 des poids d'un modèle de réseau de neurones.

    Paramètres :
    - model (torch.nn.Module) : Le modèle dont la norme L1 des poids doit être calculée.
    - checkpoint_path (str) : Chemin vers le point de contrôle (checkpoint) contenant les poids du modèle.
    - print_norm (bool) : Afficher ou non la norme L1 calculée.

    Renvoie :
    - total_norm (float) : Norme L1 totale des poids du modèle.
    """
    total_norm = 0
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    poids_couche1 = model.output_layer.weight
    somme_poids_absolus = torch.sum(torch.abs(poids_couche1))

    print(somme_poids_absolus)
    for param in model.parameters():
        total_norm += param.abs().sum().item()
    
    if print_norm:
        print(f'Norme L1 des poids du réseau: {somme_poids_absolus:.4f}')

    if somme_poids_absolus < 1:
        return 1
    return somme_poids_absolus.detach().numpy()
