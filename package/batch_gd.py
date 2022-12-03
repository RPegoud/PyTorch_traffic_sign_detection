from datetime import datetime
import numpy as np
import torch

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs: int = 10):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, test_losses = np.zeros(epochs), np.zeros(epochs)

    for it in range(epochs):  # iterate over epochs

        t0 = datetime.now()
        model.train()
        train_loss = []

        for inputs, targets in train_loader:  # iterate over batches
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
    
        model.eval()
        test_loss = []

        for inputs, targets in test_loader:  # iterate over batches
            inputs, targets = inputs.to(
                device), targets.to(device)  # convert the targets to the new classes
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        train_losses[it], test_losses[it] = train_loss,  test_loss
        dt = datetime.now() - t0

        print(f'Epoch {it+1} / {epochs}: Train Loss : {train_loss:.4f}, Test Loss : {test_loss:.4f}, duration: {dt}')

    return train_losses, test_losses