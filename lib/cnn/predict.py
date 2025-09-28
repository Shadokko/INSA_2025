import numpy as np
import torch
from torch.autograd import Variable


def predict(model, test, batch_size=128, validation_size=-1, n_workers=8, device_id=0):
    """
        Give the prediction of the model on a test set
        :param model: the model
        :param test_loader: the test set loader
        :param validation_size: number of occurrences for the validation
    """
    test_loader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=batch_size, num_workers=n_workers)
    with torch.no_grad():
        total = 0
        model.eval()

        y_preds = []
        y_labels = []
        for data in test_loader:

            inputs, labels = data
            inputs = Variable(inputs.to(torch.device('cuda:'+str(device_id)))) if torch.cuda.is_available() else Variable(inputs)
            outputs = model(inputs)

            y_preds.extend(outputs.data.tolist())
            y_labels.extend(labels.tolist())

            if total >= validation_size != -1:
                break

        predictions, labels = np.asarray(y_preds), np.asarray(y_labels)

    return predictions, labels
