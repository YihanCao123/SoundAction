from pyexpat import model
import numpy as np
import logging
from sklearn import metrics
from model.utils import forward

from data.utils import get_filename
from params import train_config

def calculate_acc(y_true, y_pred):
    N = y_true.shape[0]
    acc = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1)) / N
    return acc


class Eva:
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader):
        # Forward
        output_dict = forward(
            model=self.model,
            generator=data_loader,
            return_target=True
        )

        clipwise_output = output_dict['clipwise_output']
        target = output_dict['target']

        cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), labels=None)
        acc = calculate_acc(target, clipwise_output)

        statistics = {'accuracy': acc, 'confusion_matrix': cm}

        return statistics