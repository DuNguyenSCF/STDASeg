# Import packages and libraries|
import os
import glob
import re
import sys
import csv
import shutil
import random
import functools

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from tqdm import tqdm as tqdm
from torch import Tensor
from torch.autograd import Variable
from torch import einsum


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(10, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    


# Preprocessing input
def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

# Training classes
#-----------------------------------------------Training------------------------------------
## Meter
# class Meter(object):
#     """Meters provide a way to keep track of important statistics in an online manner.
#     This class is abstract, but provides a standard interface for all meters to follow.
#     """

#     def reset(self):
#         """Reset the meter to default settings."""
#         pass

#     def add(self, value):
#         """Log a new value to the meter
#         Args:
#             value: Next result to include.
#         """
#         pass

#     def value(self):
#         """Get the value of the meter in the current state."""
#         pass

# class AverageValueMeter(Meter):
#     def __init__(self):
#         super(AverageValueMeter, self).__init__()
#         self.reset()
#         self.val = 0

#     def add(self, value, n=1):
#         self.val = value
#         self.sum += value
#         self.var += value * value
#         self.n += n

#         if self.n == 0:
#             self.mean, self.std = np.nan, np.nan
#         elif self.n == 1:
#             self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
#             self.std = np.inf
#             self.mean_old = self.mean
#             self.m_s = 0.0
#         else:
#             self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
#             self.m_s += (value - self.mean_old) * (value - self.mean)
#             self.mean_old = self.mean
#             self.std = np.sqrt(self.m_s / (self.n - 1.0))

#     def value(self):
#         return self.mean, self.std

#     def reset(self):
#         self.n = 0
#         self.sum = 0.0
#         self.var = 0.0
#         self.val = 0.0
#         self.mean = np.nan
#         self.mean_old = 0.0
#         self.m_s = 0.0
#         self.std = np.nan

## Training params
# class Epoch:
#     def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
#         self.model = model
#         self.loss = loss
#         self.metrics = metrics
#         self.stage_name = stage_name
#         self.verbose = verbose
#         self.device = device

#         self._to_device()

#     def _to_device(self):
#         self.model.to(self.device)
#         self.loss.to(self.device)
#         for metric in self.metrics:
#             metric.to(self.device)

#     def _format_logs(self, logs):
#         str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
#         s = ", ".join(str_logs)
#         return s

#     def batch_update(self, x, y):
#         raise NotImplementedError

#     def on_epoch_start(self):
#         pass

#     def run(self, dataloader):

#         self.on_epoch_start()
#         #write csv file
#         # fieldnames = ['loss_logs', 'metrics_logs']
#         # with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
#         #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         #     writer.writeheader()

#         logs = {}
#         loss_meter = AverageValueMeter()
#         metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

#         with tqdm(
#             dataloader,
#             desc=self.stage_name,
#             file=sys.stdout,
#             disable=not (self.verbose),
#         ) as iterator:
#             for x, y in iterator:
#                 x, y = x.to(self.device), y.to(self.device)
#                 loss, y_pred = self.batch_update(x, y)

#                 # update loss logs
#                 loss_value = loss.cpu().detach().numpy()
#                 loss_meter.add(loss_value)
#                 loss_logs = {self.loss.__name__: loss_meter.mean}
#                 logs.update(loss_logs)

#                 # update metrics logs
#                 for metric_fn in self.metrics:
#                     metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
#                     metrics_meters[metric_fn.__name__].add(metric_value)
                    
#                 # metrics_logs = {k: np.mean(v.mean) for k, v in metrics_meters.items()}
#                 metrics_logs = {}
#                 for k, v in metrics_meters.items():
#                     if k == 'AIU_score':
#                         metrics_logs.update({k: np.mean(v.mean)})
#                     else:
#                         metrics_logs.update({k: np.max(v.mean)})
#                 logs.update(metrics_logs)

#                 if self.verbose:
#                     s = self._format_logs(logs)
#                     iterator.set_postfix_str(s)

#         return logs

class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
    
class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction