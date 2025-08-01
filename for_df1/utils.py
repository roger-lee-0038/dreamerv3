import numpy as np
import time
from concurrent import futures

import math
from typing import Optional

import torch

def retransform(x_real:np.ndarray, bounds_lower, bounds_upper):
    mask = bounds_lower != bounds_upper # == is possible if update_bounds
    x_temp = x_real
    x_temp[mask] = (x_real[mask]-bounds_lower[mask])/(bounds_upper[mask]-bounds_lower[mask])
    x_temp[~mask] = 0
    x_01 = x_temp.clip(0,1)
    return x_01

def transform(x_01:np.ndarray, bounds_lower, bounds_upper):
    x_real = x_01 * (bounds_upper-bounds_lower) + bounds_lower
    return x_real

class Func: # avoid closure for multiprocessing
    def __init__(self, obj_func, func_sleep):
        self.obj_func = obj_func
        self.func_sleep = func_sleep
    def __call__(self, x_real: np.ndarray) -> float:
        y = self.obj_func(x_real)
        time.sleep(self.func_sleep)
        return y

def run_funcs(func, x_real_chunk, num_workers):
    # assert len(x_real_chunk) == num_workers
    results_list = []
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        running_jobs = []
        finished_jobs = []
        for index, x_real in enumerate(x_real_chunk):
            running_jobs.append(
                (index, x_real, executor.submit(func, x_real))
            )
        while running_jobs or finished_jobs:
            if finished_jobs:
                for index, x_real, job in finished_jobs:
                    value = job.result()
                    results_list.append((index, x_real, value))
                finished_jobs = []
            tmp_runnings, tmp_finished = [], []
            for index, x_real, job in running_jobs:
                (tmp_finished if job.done() else tmp_runnings).append((index, x_real, job))
            running_jobs, finished_jobs = tmp_runnings, tmp_finished
    fx_chunk = [results[2] for results in sorted(results_list, key=lambda i:i[0])]
    return fx_chunk

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            #print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                #print('INFO: Early stopping')
                self.early_stop = True
