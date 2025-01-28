from PixelCNN_MNIST_model import PixelCNN_MNIST,device
import torch
import numpy as np
from torch.autograd import Variable

def train_loop(dataloader, model, loss_fn, optimizer,best_loss):
    """Function used to train a network. Performs one epoch.
    
    :parameter dataloader: data that will be used to train the network.
    :param model: the architecture for the model to be trained.
    :param loss_fn: the loss function chosen for the training.
    :param optimizer: the optimizer chosen for the training.
    :param best_loss: the minimal loss recorded during training.

    :returns: two elements : the mean loss during the whole epoch and the minimum found
    :rtype: tuple of two elements
    """
    size = len(dataloader.dataset)
    model = model.to(device)
    list_loss = []
    PATH = 'pages/training_weight/mnist_user_train_standby.pth'
    for batch, (X, y) in enumerate(dataloader):
        target = X[:,:,:,:]
        X = X.to(device)
        target = target.to(device)
        # Compute prediction and loss
        pred = model(X)
        pred = pred[:,:,:,:] 
        #print(pred)
        
        loss = loss_fn(pred,target)
        list_loss.append(loss.item())
        # Backpropagation (always in three steps)
        optimizer.zero_grad() # a toujours mettre pour pas accumuler les gradients
        loss.backward()
        optimizer.step()
        if best_loss > loss.item(): #comparing the loss to the registered minimum
            best_loss = loss.item()
            torch.save(model.state_dict(), PATH) #saving the associated model if it is a new minimum
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return (np.mean(list_loss),best_loss)