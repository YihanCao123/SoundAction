from pyexpat import model
import numpy as np
import logging
from sklearn import metrics
from model.utils import forward

from data.utils import get_filename
from params import train_config

def calculate_acc(y_true, y_pred):
    print('y_true', np.argmax(y_true, axis=-1))
    print('y_pred',np.argmax(y_pred, axis=-1))
    N = y_true.shape[0]
    acc = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1)) / N
    return acc

def numpy_decrible(arr):
    # measures of dispersion
    min_ = np.amin(arr)
    max_ = np.amax(arr)
    range_ = np.ptp(arr)
    variance = np.var(arr)
    sd = np.std(arr)
    mean_ = np.mean(arr)
    
    print("Array =", arr)
    print("Measures of Dispersion")
    print("Minimum =", min_)
    print("Maximum =", max_)
    print("Range =", range_)
    print("Variance =", variance)
    print("Standard Deviation =", sd)
    print("Mean =", mean_)

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

        output = output_dict['predict_target']
        target = output_dict['target']

        print('Tutput summary:')
        numpy_decrible(output)
        print('Target summary:')
        numpy_decrible(target)


        #  np.mean((target - (output > 0)) == 0)
        
        statistics = {'accuracy': np.mean(np.abs(output - target) < 0.25), 'confusion_matrix': None}

        return statistics