from datetime import datetime
import numpy as np
import torch
from tqdm.auto import tqdm


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def batch_gd(model, criterion, optimizer, train_loader, test_loader, early_stopping:bool = False, \
    early_stopper:EarlyStopping = None, epochs: int = 10):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, test_losses = np.zeros(epochs), np.zeros(epochs)

    for it in range(epochs):  # iterate over epochs

        # ----- Training -----

        t0 = datetime.now()
        model.train()
        train_loss = []

        for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):  # iterate over batches
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)  # convert the targets to the new classes
            optimizer.zero_grad() # reset the optimizer gradient between steps
            
            # forward pass
            outputs = model(inputs)  
            loss = criterion(outputs, targets)

            # backward pass
            loss.backward() # compute the gradient
            optimizer.step() # perform a step of gradient descent

            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
    

        # ----- Eval -----

        model.eval()
        test_loss = []

        for idx, data in tqdm(enumerate(test_loader), total=len(test_loader)):  # iterate over batches
            inputs, targets = data  # iterate over batches
            inputs, targets = inputs.to(
                device), targets.to(device)  # convert the targets to the new classes
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        train_losses[it], test_losses[it] = train_loss,  test_loss
        dt = datetime.now() - t0

        if early_stopping:
            if early_stopper.early_stop(test_loss):
                print(f'Stopped at epoch: {it+1} with Train Loss : {train_loss:.4f}, Test Loss : {test_loss:.4f}')
                return train_losses, test_losses
        
        print(f'Epoch {it+1} / {epochs}: Train Loss : {train_loss:.4f}, Test Loss : {test_loss:.4f}, duration: {dt}')

    return train_losses, test_losses