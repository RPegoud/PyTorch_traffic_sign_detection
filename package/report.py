import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm

def report(model, loader, train_losses, test_losses):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test_loss')
    plt.legend()
    plt.plot

    p_test, y_test = np.array([]), np.array([])
    X_test = []
    model.eval()

    for idx, data in tqdm(enumerate(loader), total=len(loader)):  # iterate over batches
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        X_test.append(inputs)
        p_test = np.concatenate((p_test, predictions.cpu().numpy()))
        y_test = np.concatenate((y_test, targets.cpu()))

    print(classification_report(y_test, p_test))
    cm = confusion_matrix(y_test, p_test)
    plt.figure(figsize=(15,15))
    sns.heatmap(cm, cmap='Blues', annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True label')
    plt.show()

    return X_test, p_test, y_test
