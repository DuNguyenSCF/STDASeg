import functools
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch import einsum
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    pass

#------------------------------------Metrics---------------------------------------
def _get_channels(*tensors, ignore_channels = None):
    if ignore_channels is None:
        return tensors
    else:
        channels = [channel for channel in range(tensors[0].shape[1]) if channel not in ignore_channels]
        tensors = [torch.index_select(tensor, dim=1, index=torch.tensor(channels).to(tensor.device)) for tensor in tensors]
        return tensors

def _threshold(pred, threshold=None):
    if threshold is not None:
        return (pred > threshold).type(pred.dtype)
    else:
        return pred
    
def _threshold_for_bitwise(pred, threshold=None):
    if threshold is not None:
        return (pred > threshold)
    else:
        return pred

# def _iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
#     """Calculate Intersection over Union between ground truth and prediction
#     Args:
#         pr (torch.Tensor): predicted tensor
#         gt (torch.Tensor):  ground truth tensor
#         eps (float): epsilon to avoid zero division
#         threshold: threshold for outputs binarization
#     Returns:
#         float: IoU (Jaccard) score
#     """

#     pr = _threshold(pr, threshold=threshold)
#     pr, gt = _get_channels(pr, gt, ignore_channels=ignore_channels)

#     pr_flat = pr.view(-1,)
#     gt_flat = gt.view(-1,).float()

#     tp = torch.sum(pr_flat * gt_flat)  # TP
#     fp = torch.sum(pr_flat * (1 - gt_flat))  # FP
#     fn = torch.sum((1 - pr_flat) * gt_flat)  # FN
#     tn = torch.sum((1 - pr_flat) * (1 - gt_flat))  # TN
    
#     if torch.sum(gt_flat) == 0:
#         iou = (tp + eps) / (tp + fp + fn + eps)
#     else:
#         iou = (tp) / (tp + fp + fn + eps)
#     return iou

def _iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    
    pr, gt = _get_channels(pr, gt, ignore_channels=ignore_channels)
    pr = pr.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    pr = _threshold_for_bitwise(pr, threshold=threshold)
    gt = _threshold_for_bitwise(gt, threshold=threshold)

    intersection = (pr & gt).sum(axis=(1,2,3))
    union = (pr | gt).sum(axis=(1,2,3))
    
    # mod = (1-gt.sum(axis=(1,2,3)).astype(np.bool_))*eps
    # iou = (intersection + mod) / (union + eps)
           
    if np.sum(gt) == 0:
        iou = (intersection + eps) / (union + eps)
    else:
        iou = (intersection) / (union + eps)
    return iou.mean()

def _aiu(pr, gt, eps=1e-7, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
    Returns:
        float: IoU (Jaccard) score
    """
    gt = gt*255
    pr = pr*255
    gt_num = torch.tensor(torch.numel(gt[gt>128]))
    
    pp = pr[gt>128]
    nn = pr[gt<=128]
    
    pp_hist = torch.histc(pp, bins=255, min=0, max=255, out=None) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist = torch.histc(nn, bins=255, min=0, max=255, out=None)
    
    pp_hist_flip = torch.flipud(pp_hist)
    nn_hist_flip = torch.flipud(nn_hist)
    
    pp_hist_cum = torch.cumsum(pp_hist_flip, dim=0)
    nn_hist_cum = torch.cumsum(nn_hist_flip, dim=0)
    
    if gt_num == 0:
        aiu = pp_hist_cum+eps/ (gt_num + nn_hist_cum+eps)
    else:
        aiu = pp_hist_cum / (gt_num + nn_hist_cum+eps)
    
    # aiu = pp_hist_cum / (gt_num + nn_hist_cum+eps)
    # aiu = torch.mean(iu)
    
    
    return aiu



# def _dice_score(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
#     """Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
#     calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks
#     Args:
#         pr (torch.Tensor): predicted tensor
#         gt (torch.Tensor):  ground truth tensor
#         eps (float): epsilon to avoid zero division
#         threshold: threshold for outputs binarization
#     Returns:
#         float: Dice score
#     """

#     pr = _threshold(pr, threshold=threshold)
#     pr, gt = _get_channels(pr, gt, ignore_channels=ignore_channels)

#     pr_flat = pr.view(-1,)
#     gt_flat = gt.view(-1,).float()

#     tp = torch.sum(pr_flat * gt_flat)  # TP
#     fp = torch.sum(pr_flat * (1 - gt_flat))  # FP
#     fn = torch.sum((1 - pr_flat) * gt_flat)  # FN
#     tn = torch.sum((1 - pr_flat) * (1 - gt_flat))  # TN
    
#     if torch.sum(gt_flat) == 0:
#         dice_score = (2*tp + eps) / (2*tp + fp + fn + eps)
#     else:
#         dice_score = (2*tp) / (2*tp + fp + fn + eps)
#     return dice_score

def _dice_score(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: Dice score
    """

    # pr = _threshold(pr, threshold=threshold)
    pr, gt = _get_channels(pr, gt, ignore_channels=ignore_channels)
    pr = pr.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    pr = _threshold_for_bitwise(pr, threshold=threshold)
    gt = _threshold_for_bitwise(gt, threshold=threshold)

    intersection = (pr & gt).sum(axis=(1,2,3))
    
    # mod = (1-gt.sum(axis=(1,2,3)).astype(np.bool_))*eps
    # dice_score = (2*intersection + mod) / (pr.sum(axis=(1,2,3)) + gt.sum(axis=(1,2,3)) + eps)
    if np.sum(gt) == 0:
        dice_score = (2*intersection + eps) / (pr.sum(axis=(1,2,3)) + gt.sum(axis=(1,2,3)) + eps)
    else:
        dice_score = (2*intersection) / (pr.sum(axis=(1,2,3)) + gt.sum(axis=(1,2,3)) + eps)
    return dice_score.mean()

def _precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: Precision
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _get_channels(pr, gt, ignore_channels=ignore_channels)

    pr_flat = pr.view(-1,)
    gt_flat = gt.view(-1,).float()

    tp = torch.sum(pr_flat * gt_flat)  # TP
    fp = torch.sum(pr_flat * (1 - gt_flat))  # FP
    fn = torch.sum((1 - pr_flat) * gt_flat)  # FN
    tn = torch.sum((1 - pr_flat) * (1 - gt_flat))  # TN

    if torch.sum(gt_flat) == 0:
        precision = (tp + eps) / (tp + fp + eps)
    else:
        precision = (tp) / (tp + fp + eps)
    return precision

def _recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: Recall
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _get_channels(pr, gt, ignore_channels=ignore_channels)

    pr_flat = pr.view(-1,)
    gt_flat = gt.view(-1,).float()

    tp = torch.sum(pr_flat * gt_flat)  # TP
    fp = torch.sum(pr_flat * (1 - gt_flat))  # FP
    fn = torch.sum((1 - pr_flat) * gt_flat)  # FN
    tn = torch.sum((1 - pr_flat) * (1 - gt_flat))  # TN

    if torch.sum(gt_flat) == 0:
        recall = (tp + eps) / (tp + fn + eps)
    else:
        recall = (tp) / (tp + fn + eps)
    return recall

def _specificity(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Specificity is the Ratio of true negatives to total negatives in the data.
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: Specificity
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _get_channels(pr, gt, ignore_channels=ignore_channels)

    pr_flat = pr.view(-1,)
    gt_flat = gt.view(-1,).float()

    tp = torch.sum(pr_flat * gt_flat)  # TP
    fp = torch.sum(pr_flat * (1 - gt_flat))  # FP
    fn = torch.sum((1 - pr_flat) * gt_flat)  # FN
    tn = torch.sum((1 - pr_flat) * (1 - gt_flat))  # TN

    if torch.sum(gt_flat) == 0:
        specificity = (tn + eps) / (tn + fp + eps)
    else:
        specificity = (tn) / (tn + fp + eps)
    return specificity

def _f1_score(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Same with Dice score
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F1 score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _get_channels(pr, gt, ignore_channels=ignore_channels)

    pr_flat = pr.view(-1,)
    gt_flat = gt.view(-1,).float()

    tp = torch.sum(pr_flat * gt_flat)  # TP
    fp = torch.sum(pr_flat * (1 - gt_flat))  # FP
    fn = torch.sum((1 - pr_flat) * gt_flat)  # FN
    tn = torch.sum((1 - pr_flat) * (1 - gt_flat))  # TN

    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    f1 = 2* (precision*recall + eps) / (precision + recall + eps)
    return f1

def _auc_score(pr, gt, eps=1e-7):
    """Same with Dice score
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: auc score
    """

#     pr = _threshold(pr, threshold=threshold)
#     pr, gt = _get_channels(pr, gt, ignore_channels=ignore_channels)

    pr_flat = pr.view(-1,).cpu().detach().numpy()
    gt_flat = gt.view(-1,).float().cpu().detach().numpy()

    precision, recall, _ = precision_recall_curve(gt_flat, pr_flat)
    auc_score = torch.tensor(auc(recall, precision))
    return auc_score

def _ods(pr, gt, eps=1e-7, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
    Returns:
        float: IoU (Jaccard) score
    """
    gt = gt*255
    pr = pr*255
    gt_num = torch.tensor(torch.numel(gt[gt>128]))
    
    pp = pr[gt>128]
    nn = pr[gt<=128]
    
    pp_hist = torch.histc(pp, bins=255, min=0, max=255, out=None) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist = torch.histc(nn, bins=255, min=0, max=255, out=None)
    
    pp_hist_flip = torch.flipud(pp_hist)
    nn_hist_flip = torch.flipud(nn_hist)
    
    pp_hist_cum = torch.cumsum(pp_hist_flip, dim=0)
    nn_hist_cum = torch.cumsum(nn_hist_flip, dim=0)
    
    if gt_num == 0:
        precision = pp_hist_cum+eps/(pp_hist_cum + nn_hist_cum+eps) #TP/(TP+FP)
        recall = pp_hist_cum+eps/(gt_num+eps) #TP/(TP+FN)
    else:
        precision = pp_hist_cum/(pp_hist_cum + nn_hist_cum+eps) #TP/(TP+FP)
        recall = pp_hist_cum/(gt_num+eps) #TP/(TP+FN)
        
    # precision = pp_hist_cum/(pp_hist_cum + nn_hist_cum+eps) #TP/(TP+FP)
    # recall = pp_hist_cum/(gt_num+eps) #TP/(TP+FN)
    
    precision[torch.isnan(precision)]= 0.0
    recall[torch.isnan(recall)] = 0.0
    
    ods = 2*precision*recall / (precision + recall + eps) # F1 score
    # aiu = torch.mean(iu)
    
    
    return ods

class IoU(Metric):
    __name__ = "IoU_score"

    def __init__(self, eps=1e-7, threshold=None, from_logits=True, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.from_logits = from_logits
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        if self.from_logits:
            y_pr = F.logsigmoid(y_pr).exp()
        return _iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
    
class AIU(Metric):
    __name__ = "AIU_score"

    def __init__(self, eps=1e-7, from_logits=True, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.from_logits = from_logits
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        if self.from_logits:
            y_pr = F.logsigmoid(y_pr).exp()
        return _aiu(
            y_pr,
            y_gt,
            eps=self.eps,
            ignore_channels=self.ignore_channels,
        )

class AUC(Metric):
    __name__ = "AUC_score"

    def __init__(self, eps=1e-7, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.from_logits = from_logits

    def forward(self, y_pr, y_gt):
        if self.from_logits:
            y_pr = F.logsigmoid(y_pr).exp()
        return _auc_score(
            y_pr,
            y_gt,
            eps=self.eps,
        )
    
    
class Dice_Score(Metric):
    __name__ = "Dice_Score"

    def __init__(self, eps=1e-7, threshold=None, from_logits=True, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.from_logits = from_logits
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        if self.from_logits:
            y_pr = F.logsigmoid(y_pr).exp()
        return _dice_score(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
    
class Precision(Metric):
    __name__ = "Precision"

    def __init__(self, eps=1e-7, threshold=None, from_logits=True, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.from_logits = from_logits
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        if self.from_logits:
            y_pr = F.logsigmoid(y_pr).exp()
        return _precision(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

class Recall(Metric):
    __name__ = "Recall"

    def __init__(self, eps=1e-7, threshold=None, from_logits=True, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.from_logits = from_logits
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        if self.from_logits:
            y_pr = F.logsigmoid(y_pr).exp()
        return _recall(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
    
class Specificity(Metric):
    __name__ = "Specificity"

    def __init__(self, eps=1e-7, threshold=None, from_logits=True, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.from_logits = from_logits
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        if self.from_logits:
            y_pr = F.logsigmoid(y_pr).exp()
        return _specificity(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

class F1_Score(Metric):
    __name__ = "F1_Score"

    def __init__(self, eps=1e-7, threshold=None, from_logits=True, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.from_logits = from_logits
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        if self.from_logits:
            y_pr = F.logsigmoid(y_pr).exp()
        return _f1_score(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
    
    
class ODS(Metric):
    __name__ = "ODS_score"

    def __init__(self, eps=1e-7, from_logits=True, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        # self.threshold = threshold
        self.from_logits = from_logits
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        if self.from_logits:
            y_pr = F.logsigmoid(y_pr).exp()
        return _ods(
            y_pr,
            y_gt,
            eps=self.eps,
            # threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

    
# class EarlyStopping():
# """
# Early stopping to stop the training when the loss does not improve after
# certain epochs.
# """
# def __init__(self, patience=20, min_delta=0):
#     """
#     :param patience: how many epochs to wait before stopping when loss is
#            not improving
#     :param min_delta: minimum difference between new loss and old loss for
#            new loss to be considered as an improvement
#     """
#     self.patience = patience
#     self.min_delta = min_delta
#     self.counter = 0
#     self.best_loss = None
#     self.early_stop = False
# def __call__(self, val_loss):
#     print('best_loss:', self.best_loss)
#     print('val_loss:', val_loss)
#     if self.best_loss == None:
#         self.best_loss = val_loss
#     elif self.best_loss - val_loss > self.min_delta:
#         self.best_loss = val_loss
#         # reset counter if validation loss improves
#         self.counter = 0
#     elif self.best_loss - val_loss < self.min_delta:
#         self.counter += 1
#         print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
#         if self.counter >= self.patience:
#             print('INFO: Early stopping')
#             self.early_stop = True

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
    def __call__(self, metric):
        print('best_metric:', self.best_metric)
        print('val_metric:', metric)
        if self.best_metric == None:
            self.best_metric = metric
        elif self.best_metric - metric < self.min_delta:
            self.best_metric = metric
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_metric - metric > self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


# class EarlyStopping():
#     """
#     Early stopping to stop the training when the loss does not improve after
#     certain epochs.
#     """
#     def __init__(self, patience=20, min_delta=0):
#         """
#         :param patience: how many epochs to wait before stopping when loss is
#                not improving
#         :param min_delta: minimum difference between new loss and old loss for
#                new loss to be considered as an improvement
#         """
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_metric1 = None
#         self.best_metric2 = None
#         self.early_stop = False
#     def __call__(self, metric1, metric2):
#         print('best_metric1:', self.best_metric1)
#         print('best_metric2:', self.best_metric2)
#         print('val_metric1:', metric1)
#         print('val_metric2:', metric2)
#         if (self.best_metric1 == None) and (self.best_metric2 == None):
#             self.best_metric1 = metric1
#             self.best_metric2 = metric2
#         elif self.best_metric1 - metric1 < self.min_delta or self.best_metric2 - metric2 < self.min_delta:
#             self.best_metric1 = metric1
#             self.best_metric2 = metric2
#             # reset counter if validation loss improves
#             self.counter = 0
#         elif self.best_metric1 - metric1 > self.min_delta and self.best_metric2 - metric2 > self.min_delta:
#             self.counter += 1
#             print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
#             if self.counter >= self.patience:
#                 print('INFO: Early stopping')
#                 self.early_stop = True