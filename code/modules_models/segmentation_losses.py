from __future__ import print_function, division
from typing import Optional, Union

import torch
import re

from functools import partial
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable
from torch import einsum

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

import numpy as np
import math

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

#-----------------------------------------------------------------------------------------
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


class Loss(BaseObject):
    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other):
        return self.__mul__(other)


class SumOfLosses(Loss):
    def __init__(self, l1, l2, a=1, b=1):
        name = "{}_{}".format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2
        self.a = a
        self.b = b

    def __call__(self, *inputs):
        return self.l1.forward(*inputs)*self.a + self.l2.forward(*inputs)*self.b


class MultipliedLoss(Loss):
    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split("+")) > 1:
            name = "{} * ({})".format(multiplier, loss.__name__)
        else:
            name = "{} * {}".format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs):
        return self.multiplier * self.loss.forward(*inputs)
    
    
    
###******************************************************************************************************************************
###******************************************************************************************************************************
###******************************************************************************************************************************
################################################LOSSES FUNCTION##################################################################

    
class BCELoss_with_logits(nn.Module):
    __name__= "BCELoss_with_logits"
    def __init__(self, reduction='mean', truncate=False):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError('"{}" is not a valid mode for reduction. Only "mean"'
                             'and "sum" are allowed.'.format(rediction))
        self.reduction = reduction
        self.truncate = truncate
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        # classical without sigmoid
        # but you need to cut off large negative logits, otherwise it will be -inf -> nan 
        if self.truncate:
            outputs = torch.sigmoid(outputs)
        outputs = outputs.float()
        labels = labels.float()
        # print(torch.min(outputs))
        bce = outputs - labels * outputs + torch.log(1 + torch.exp(-outputs))
        if self.reduction == 'mean':
            return torch.mean(bce)
        elif self.reduction == 'sum':
            return torch.sum(bce)
        
class BCELoss_classic(nn.Module):
    __name__= "BCE_Loss"
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError('"{}" is not a valid mode for reduction. Only "mean"'
                             'and "sum" are allowed.'.format(reduction))
        self.reduction = reduction
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs = torch.clamp(outputs, 0, 1)
        outputs = F.logsigmoid(outputs).exp()
        bce = labels * torch.log(outputs + 1e-7) + (1 - labels) * torch.log(1 - outputs + 1e-7)
        if self.reduction == 'mean':
            return -torch.mean(bce)
        elif self.reduction == 'sum':
            return -torch.sum(bce)
    
def focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
    ignore_index=None,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type_as(output)

    p = F.logsigmoid(output).exp()
    ce_loss = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - p_t).pow(gamma)
    else:
        focal_term = ((1.0 - p_t) / reduced_threshold).pow(gamma)
        focal_term = torch.masked_fill(focal_term, p_t < reduced_threshold, 1)

    loss = focal_term * ce_loss

    if alpha is not None:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss *= alpha_t

    if ignore_index is not None:
        ignore_mask = target.eq(ignore_index)
        loss = torch.masked_fill(loss, ignore_mask, 0)
        if normalized:
            focal_term = torch.masked_fill(focal_term, ignore_mask, 0)

    if normalized:
        norm_factor = focal_term.sum(dtype=torch.float32).clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum(dtype=torch.float32)
    if reduction == "batchwise_mean":
        loss = loss.sum(dim=0, dtype=torch.float32)

    return loss

    
class FocalLoss_b(_Loss):
    
    __name__= "FocalLoss_bloss"


    def __init__(
        self,
        alpha=0.25,
        gamma: float = 2.0,
        ignore_index=None,
        reduction="mean",
        normalized=False,
        reduced_threshold=None,
    ):
        """
        :param alpha: Prior probability of having positive value in target.
        :param gamma: Power factor for dampening weight (focal strenght).
        :param ignore_index: If not None, targets may contain values to be ignored.
        Target values equal to ignore_index will be ignored from loss computation.
        :param reduced: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
        :param threshold:
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
            ignore_index=ignore_index,
        )

    def forward(self, output, target):
        """Compute focal loss for binary classification problem.
        label_input shape and label target must be the same
        """
        loss = self.focal_loss_fn(output, target)
        return loss
    
class mMFW_BCEWithLogitLoss(nn.Module):

    __name__ = "mMFW_BCEWithLogitLoss"

    """
    Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
    Weighted Binary Cross Entropy.
    `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1. 
    To decrease the number of false positives, set β<1. 
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, pos_weight=1.0, neg_weight=1.0, ignore_index=None, reduction='mean', smooth = None):
        super(mMFW_BCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        pos_weight = float(pos_weight)
        neg_weight = float(neg_weight)
        self.pos_weight = pos_weight # how to set weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = F.logsigmoid(output).exp() # should be replace with logsigmoid function
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

        # loss = self.bce(output, target)
        loss = -self.pos_weight * target.mul(torch.log(output)) - self.neg_weight*((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss
    
class mIFW_BCEWithLogitLoss(nn.Module):

    __name__ = "mIFW_BCEWithLogitLoss"

    """
    Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
    Weighted Binary Cross Entropy.
    `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1. 
    To decrease the number of false positives, set β<1. 
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, pos_weight=1.0, neg_weight=1.0, ignore_index=None, reduction='mean', smooth = None):
        super(mIFW_BCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        pos_weight = float(pos_weight)
        neg_weight = float(neg_weight)
        self.pos_weight = pos_weight # how to set weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = F.logsigmoid(output).exp() # should be replace with logsigmoid function
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

        # loss = self.bce(output, target)
        loss = -self.pos_weight * target.mul(torch.log(output)) - self.neg_weight*((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss
    
class IFW_BCE(nn.Module):

    __name__ = "IFW_BCE"

    """
    Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
    Weighted Binary Cross Entropy.
    `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1. 
    To decrease the number of false positives, set β<1. 
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, ignore_index=None, reduction='mean', smooth = None):
        super(IFW_BCE, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)
         

        output = F.logsigmoid(output).exp() # should be replace with logsigmoid function
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
        pos_target: Tensor = target.eq(1).sum()
        neg_target: Tensor = target.eq(0).sum()
        num_target = pos_target + neg_target
        pos_weight = (num_target + eps) / (pos_target + eps)
        neg_weight = (num_target + eps) / (neg_target + eps)

        # loss = self.bce(output, target)
        loss = -pos_weight * target.mul(torch.log(output)) - neg_weight*((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss    


    
# class RecallCE(nn.Module):
#     __name__= "RecallCE"
#     def __init__(self, reduction='mean'):
#         super().__init__()
#         if reduction not in ('mean', 'sum'):
#             raise ValueError('"{}" is not a valid mode for reduction. Only "mean"'
#                              'and "sum" are allowed.'.format(rediction))
#         self.reduction = reduction
        
#     def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
#         # outputs = torch.clamp(outputs, 0, 1)
#         outputs = F.logsigmoid(outputs).exp()
#         Nc = labels.sum()
#         Nb = (1-labels).sum()
#         if Nc == 0:
#             weight = 1
#         else:
#             r = Nb / Nc
#             tp = (labels * outputs).sum()
#             fn = (labels * (1 - outputs)).sum()
#             # weight = (fn / (fn + tp + 1e-7))*Nc
#             weight = (fn / (fn + tp + 1e-7))*torch.log(r)
#         # bce = labels * torch.log(outputs + 1e-7) + (1 - labels) * torch.log(1 - outputs + 1e-7)
#         recallce = weight * labels * torch.log(outputs + 1e-7) + (1 - labels) * torch.log(1 - outputs + 1e-7)
#         # recallce = weight * bce
#         if self.reduction == 'mean':
#             return -torch.mean(recallce)
#         elif self.reduction == 'sum':
#             return -torch.sum(recallce)
        
class RecallCE(nn.Module):
    __name__= "RecallCE"
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError('"{}" is not a valid mode for reduction. Only "mean"'
                             'and "sum" are allowed.'.format(rediction))
        self.reduction = reduction
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs = torch.clamp(outputs, 0, 1)
        outputs = F.logsigmoid(outputs).exp()
        
        tp_c = (labels * outputs).sum()
        fn_c = (labels * (1 - outputs)).sum()
        # weight = (fn / (fn + tp + 1e-7))*Nc
        weight_c = (fn_c + 1e-7) / (fn_c + tp_c + 1e-7)
        
        # tp_b = ((1-labels) * (1-outputs)).sum()
        # fn_b = ((1-labels) * (outputs)).sum()
        # # weight = (fn / (fn + tp + 1e-7))*Nc
        # weight_b = (fn_b + 1e-7) / (fn_b + tp_b + 1e-7)
        
        # bce = labels * torch.log(outputs + 1e-7) + (1 - labels) * torch.log(1 - outputs + 1e-7)
        recallce = weight_c * labels * torch.log(outputs+1e-7) + 1 * (1 - labels) * torch.log(1 - outputs + 1e-7)
        # recallce = weight * bce
        if self.reduction == 'mean':
            return -torch.mean(recallce)
        elif self.reduction == 'sum':
            return -torch.sum(recallce)



class Adaptive_Weighted_BCEWithLogitLoss(nn.Module):

    __name__ = "Adaptive_Weighted_BCE"


    def __init__(self, beta=0.75, ignore_index=None, reduction='mean', smooth = None):
        super(Adaptive_Weighted_BCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        self.beta = float(beta) # how to set weight
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = F.logsigmoid(output).exp() # should be replace with logsigmoid function
        # print(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
        pos_target: Tensor = target.eq(1).sum()
        neg_target: Tensor = target.eq(0).sum()
        num_target = pos_target + neg_target
        alpha = neg_target / num_target
        q_alpha = self.beta * (10**(2*alpha - 1))
        # print(pos_target, neg_target, num_target, alpha, q_alpha)

        # loss = self.bce(output, target)
        loss = -q_alpha * target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss
    
    
class Adaptive_Weighted_FTVL(nn.Module):

    __name__ = "AWFTVL"


    def __init__(self, beta=0.75, gamma = 0.25, ignore_index=None, reduction='mean', smooth = None):
        super(Adaptive_Weighted_FTVL, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        self.beta = float(beta) # how to set weight
        self.gamma = float(gamma) # how to set weight
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = F.logsigmoid(output).exp() # should be replace with logsigmoid function
        # print(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
        p_t = (1 - output) * (1 - target)
            
        pos_target: Tensor = target.eq(1).sum()
        neg_target: Tensor = target.eq(0).sum()
        num_target = pos_target + neg_target
        alpha = neg_target / num_target
        q_alpha = self.beta * (10**(2*alpha - 1))
        # print(pos_target, neg_target, num_target, alpha, q_alpha)

        # loss = self.bce(output, target)
        awbce_loss = -q_alpha * target.mul(torch.log(output)) - ((torch.pow((1.0 - p_t),(self.gamma))).mul(torch.log(1.0 - output)))
        
        
        if self.reduction == 'mean':
            awbce_loss = torch.mean(awbce_loss)
        elif self.reduction == 'sum':
            awbce_loss = torch.sum(awbce_loss)
        elif self.reduction == 'none':
            awbce_loss = awbce_loss
        else:
            raise NotImplementedError
            
        pp = (target * output).sum()
        den1 = self.beta * ((1 - target) * output).sum() # FP
        den2 = (1 - self.beta) * (target * (1 - output)).sum() # FN
        tv_index = (1 + pp) / (1 + pp + den1 + den2 + 1e-7)
        ftl = torch.pow((1 - tv_index),(1-self.gamma))
        
        loss = awbce_loss + ftl
        
        
        return loss 
    
#----------------------------------------------------------------------------------------------------------------
    
def compute_dtm(gt: Tensor) -> Tensor:
    fg_dist = np.zeros(gt.shape)
    bg_dist = np.zeros(gt.shape)
    if gt.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(gt.shape[0]): # batch size
        for c in range(dis_id, gt.shape[1]): # class_num
            posmask = gt[b][c].cpu()
            negmask = 1-posmask
            pos_dis = edt(posmask)
            neg_dis = edt(negmask)
            norm_pos_dis = (pos_dis)/(np.max(pos_dis) + 1e-7)
            norm_neg_dis = (neg_dis)/(np.max(neg_dis) + 1e-7)
            norm_pos_dis[np.isnan(norm_pos_dis)]= 0.0
            norm_neg_dis[np.isnan(norm_neg_dis)]= 0.0
            fg_dist[b][c] = norm_pos_dis
            bg_dist[b][c] = norm_neg_dis
    return torch.from_numpy(fg_dist), torch.from_numpy(bg_dist)


    
class mDTM_Weighted_BCEWithLogitLoss(nn.Module):

    __name__ = "mDTM_Weighted_BCE"

    """
    Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
    Adaptive_Weighted Binary Cross Entropy.
    `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1. 
    To decrease the number of false positives, set β<1. 
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
        super(mDTM_Weighted_BCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth = smooth
        self.device = device

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        output = F.logsigmoid(output).exp() # should be replace with logsigmoid function
        # print(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
        fg_dist, bg_dist = compute_dtm(target)
        fg_dist = (1+fg_dist).to(self.device)
        bg_dist = (1+bg_dist).to(self.device)
        # print(pos_target, neg_target, num_target, alpha, q_alpha)

        # loss = self.bce(output, target)
        loss = -fg_dist * target.mul(torch.log(output)) - bg_dist * ((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss
    
#--------------------------------------------------------------------------------------------------------------    
def compute_dpt(gt: Tensor) -> Tensor:
    dpt = np.zeros(gt.shape)
    if gt.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(gt.shape[0]): # batch size
        for c in range(dis_id, gt.shape[1]): # class_num
            posmask = gt[b][c].cpu()
            negmask = 1-posmask
            pos_dis = edt(posmask)
            neg_dis = edt(negmask)
            dis = pos_dis + neg_dis
            norm_dis = (dis)/(np.max(dis) + 1e-7)
            inverse_dis = 1 - norm_dis
            
            dpt[b][c] = inverse_dis
    return torch.from_numpy(dpt)
    
class mDPT_Weighted_BCEWithLogitLoss(nn.Module):

    __name__ = "mDPT_Weighted_BCE"

    """
    Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
    Adaptive_Weighted Binary Cross Entropy.
    `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1. 
    To decrease the number of false positives, set β<1. 
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
        super(mDPT_Weighted_BCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth = smooth
        self.device = device

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        output = F.logsigmoid(output).exp() # should be replace with logsigmoid function
        # print(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
        dpt = compute_dpt(target)
        dpt = (1+dpt).to(self.device)
        # print(pos_target, neg_target, num_target, alpha, q_alpha)

        # loss = self.bce(output, target)
        loss = -dpt*target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss
    
def compute_dtm(gt: Tensor) -> Tensor:
    fg_dist = np.zeros(gt.shape)
    bg_dist = np.zeros(gt.shape)
    if gt.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(gt.shape[0]): # batch size
        for c in range(dis_id, gt.shape[1]): # class_num
            posmask = gt[b][c].cpu()
            negmask = 1-posmask
            pos_dis = edt(posmask)
            neg_dis = edt(negmask)  
            norm_pos_dis = (pos_dis)/(np.max(pos_dis) + 1e-7)
            norm_neg_dis = (neg_dis)/(np.max(neg_dis) + 1e-7)
            norm_pos_dis[np.isnan(norm_pos_dis)]= 0.0
            norm_neg_dis[np.isnan(norm_neg_dis)]= 0.0
            fg_dist[b][c] = norm_pos_dis
            bg_dist[b][c] = norm_neg_dis
    return torch.from_numpy(fg_dist), torch.from_numpy(bg_dist)

class mDTM_Weighted_BCEWithLogitLoss(nn.Module):

    __name__ = "mDTM_Weighted_BCE"

    """
    Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
    Adaptive_Weighted Binary Cross Entropy.
    `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1. 
    To decrease the number of false positives, set β<1. 
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
        super(mDTM_Weighted_BCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth = smooth
        self.device = device

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        output = torch.sigmoid(output) # should be replace with logsigmoid function
        # print(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
        fg_dist, bg_dist = compute_dtm(target)
        fg_dist = (1+fg_dist).to(self.device)
        bg_dist = (1+bg_dist).to(self.device)
        # print(pos_target, neg_target, num_target, alpha, q_alpha)

        # loss = self.bce(output, target)
        loss = -fg_dist * target.mul(torch.log(output)) - bg_dist * ((1.0 - target).mul(torch.log(1.0 - output)))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss

class DPT_TVLloss(nn.Module):
    __name__= "DPT_TVL_Loss"

    """
    Implementation of DPT_TVL loss

    """
    def __init__(self, smooth = 1e-7, device = 'cpu'):
        super().__init__()
        self.smooth = smooth
        self.device = device
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        dpt = compute_dpt(labels).to(self.device)
        # dpt = (1+dpt).to(self.device)
        
        
        TP = (labels * outputs).sum() #TP
        FP = ((1 - labels) * outputs) # FP
        DTP_FP = ((1-dpt) * FP).sum()
        FN = (labels * (1 - outputs)) # FN
        DPT_FN = ((dpt) * FN).sum()
        DPT_TVL = TP / (TP + DTP_FP + DPT_FN + self.smooth)
        return 1 - DPT_TVL
    
class Focal_TVLloss(nn.Module):
    __name__= "Focal_TVL_Loss"

    """
    Implementation of Focal_TVL loss

    """
    def __init__(self, smooth = 1e-7, device = 'cpu', gamma:float = 1):
        super().__init__()
        self.smooth = smooth
        self.device = device
        self.gamma = gamma
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        p_t_FP = (1 - outputs) * (1 - labels)
        p_t_FN = outputs * labels
        
        TP = (labels * outputs).sum() #TP
        
        FP = ((1 - labels) * outputs) # FP
        Focal_FP = (torch.pow((1 - p_t_FP),self.gamma) * FP).sum()
        # Focal_FP = (p_t_FN * FP).sum()
        # print(Focal_FP)
        
        FN = (labels * (1 - outputs)) # FN
        Focal_FN = ((torch.pow((1 - p_t_FN),self.gamma)) * FN).sum()
        # print(Focal_FN)
        Focal_TVL = TP / (TP + Focal_FP + Focal_FN + self.smooth)
        
        return 1 - Focal_TVL
    
class mFocal_TVLloss(nn.Module):
    __name__= "mFocal_TVL_Loss"

    """
    Implementation of Focal_TVL loss with slight modification

    """
    def __init__(self, smooth = 1e-7, device = 'cpu', gamma_FP:float = 1, gamma_FN:float = 1):
        super().__init__()
        self.smooth = smooth
        self.device = device
        self.gamma_FP = gamma_FP
        self.gamma_FN = gamma_FN
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        p_t_FP = (1 - outputs) * (1 - labels)
        p_t_FN = outputs * labels
        
        TP = (labels * outputs).sum(dim=(1,2,3)) #TP
        
        FP = ((1 - labels) * outputs) # FP
        Focal_FP = (torch.pow((1 - p_t_FP),self.gamma_FP) * FP).sum(dim=(1,2,3))
        # Focal_FP = (p_t_FN * FP).sum()
        # print(Focal_FP)
        
        FN = (labels * (1 - outputs)) # FN
        Focal_FN = ((torch.pow((1 - p_t_FN),self.gamma_FN)) * FN).sum(dim=(1,2,3))
        # print(Focal_FN)
        Focal_TVL = TP / (TP + Focal_FP + Focal_FN + self.smooth)
        
        return 1 - Focal_TVL.mean()
    
    
    
class Hybrid_Focalloss(nn.Module):

    __name__ = "Hybrid_Focal_Loss"


    def __init__(self, gamma = 0.25, delta = 0.5, ignore_index=None, reduction='mean', smooth = None):
        super(Hybrid_Focalloss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        # self.beta = float(beta) # how to set weight
        self.gamma = float(gamma) # how to set weight
        self.delta = float(delta)
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = F.logsigmoid(output).exp() # should be replace with logsigmoid function
        # print(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
        p_t = (1 - output) * (1 - target)
        
        tp_c = (target * output)
        fn_c = (target * (1 - output))
        # weight = (fn / (fn + tp + 1e-7))*Nc
        focal_term_fg = (fn_c + eps) / (fn_c + tp_c + eps)
        focal_term_bg = torch.pow((1 - p_t), self.gamma)
        # focal_term_bg = 1.0
       
        focal_loss_m = -focal_term_fg * target.mul(torch.log(output)) - focal_term_bg * (1-target).mul(torch.log(1.0 - output))
        
        if self.reduction == 'mean':
            focal_loss_m = torch.mean(focal_loss_m)
        elif self.reduction == 'sum':
            focal_loss_m = torch.sum(focal_loss_m)
        elif self.reduction == 'none':
            focal_loss_m = focal_loss_m
        else:
            raise NotImplementedError
        
        pos_targets: Tensor = target.eq(1).sum()
        neg_targets: Tensor = target.eq(0).sum()
        num_targets = pos_targets + neg_targets
        if pos_targets == 0:
            alpha = 1
            beta = 1
        else:
            alpha = 1 / torch.log(neg_targets / pos_targets)
            beta = 1 - alpha
        # print('\n alpha:_',alpha,'\n')
        
        TP = (target * output).sum() #TP
        
        FP = ((1 - target) * output) # FP
        IFW_FP = (alpha * FP).sum()
        
        FN = (target * (1 - output)) # FN
        IFW_FN = (beta * FN).sum()
        
        TI_m = TP / (TP + IFW_FP + IFW_FN + eps)
        ftvl_m = torch.pow((1 - TI_m), (1 - self.gamma))
        
        hybrid_focal_loss = self.delta*focal_loss_m + (1-self.delta)*ftvl_m
        

        
        
        return hybrid_focal_loss
    

class Hybrid_Adaptive_Focalloss(nn.Module):

    __name__ = "Hybrid_Adaptive_Focal_Loss"


    def __init__(self, gamma = 2.0, ignore_index=None, reduction='mean', smooth = None):
        super(Hybrid_Adaptive_Focalloss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        self.gamma = float(gamma) # how to set weight
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        output = F.logsigmoid(output).exp() # should be replace with logsigmoid function
        # print(output)
        # avoid `nan` loss
        eps = 1e-6
        output = torch.clamp(output, min=eps, max=1.0 - eps)
        # soft label
        if self.smooth is not None:
            target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
        p_t = (1 - output) * (1 - target)
        
        pos_target: Tensor = target.eq(1).sum()
        neg_target: Tensor = target.eq(0).sum()
        num_target = pos_target + neg_target
        alpha = neg_target / num_target
        # q_alpha = (10**(2*alpha - 1))
        p_w = (neg_target+eps) / (pos_target+eps)

        # TP = (target * output)
        # FP = ((1 - target) * output) # FP
        # FN = (target * (1 - output)) # FN
        
        TP = (target * output).sum()
        FP = ((1 - target) * output).sum() # FP
        FN = (target * (1 - output)).sum() # FN
        
        # focal_term_fg = torch.pow((1 - TP / (FN + TP + eps)), self.gamma)
        beta = 1 - TP / (FN + TP + eps)
        focal_term_bg = torch.pow((1 - p_t), self.gamma)
            
       
        hybrid_adaptive_focal_loss = -beta*(p_w * target.mul(torch.log(output))) - (1 - beta) * (focal_term_bg * (1-target).mul(torch.log(1.0 - output)))
        
        # hybrid_adaptive_focal_loss = -q_alpha * (focal_term_fg * target.mul(torch.log(output))) - focal_term_bg * (1-target).mul(torch.log(1.0 - output))
        
        
        if self.reduction == 'mean':
            hybrid_adaptive_focal_loss = torch.mean(hybrid_adaptive_focal_loss)
        elif self.reduction == 'sum':
            hybrid_adaptive_focal_loss = torch.sum(hybrid_adaptive_focal_loss)
        elif self.reduction == 'none':
            hybrid_adaptive_focal_loss = hybrid_adaptive_focal_loss
        else:
            raise NotImplementedError
        
        
        return hybrid_adaptive_focal_loss

    
class IFW_TVLloss(nn.Module):
    __name__= "IFW_TVL"

    """
    Implementation of IFW_TVL loss

    """
    def __init__(self, smooth = 1e-7, gamma = 1):
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        pos_targets: Tensor = labels.eq(1).sum()
        neg_targets: Tensor = labels.eq(0).sum()
        num_targets = pos_targets + neg_targets
        if pos_targets == 0:
            alpha = 1
            beta = 1
        else:
            alpha = 1 / torch.log(neg_targets / pos_targets)
            beta = 1 - alpha
        
        TP = (labels * outputs).sum() #TP
        
        FP = ((1 - labels) * outputs) # FP
        IFW_FP = (alpha * FP).sum()
        
        FN = (labels * (1 - outputs)) # FN
        IFW_FN = (beta * FN).sum()
        
        IFW_TVL = TP / (TP + IFW_FP + IFW_FN + self.smooth)
        
        return 1 - IFW_TVL
    
    
class Focal_IFW_IoUloss(nn.Module):
    __name__= "Focal_IFW_IoU_Loss"

    """
    Implementation of Focal_IFW_IoU loss

    """
    def __init__(self, smooth = 1e-7, gamma = 0.75):
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        pos_targets: Tensor = labels.eq(1).sum()
        neg_targets: Tensor = labels.eq(0).sum()
        num_targets = pos_targets + neg_targets
        if pos_targets == 0:
            alpha = 1
            beta = 1
        else:
            alpha = pos_targets / num_targets
            beta = neg_targets / num_targets
        
        TP = (labels * outputs).sum() #TP
        
        FP = ((1 - labels) * outputs) # FP
        IFW_FP = (alpha * FP).sum()
        
        FN = (labels * (1 - outputs)) # FN
        IFW_FN = (beta * FN).sum()
        
        IFW_IoU = TP / (TP + IFW_FP + IFW_FN + self.smooth)
        
        return torch.pow((1 - IFW_IoU),self.gamma)
    

 
    
    
class DiceLoss(nn.Module):
    __name__= "mDice_Loss"
    def __init__(self, reduction='mean'):
        super().__init__()

#     # not work, why?
#     def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
#         coef = 1 / (outputs.shape[0] * outputs.shape[2] * outputs.shape[3])
#         outputs = torch.sigmoid(outputs)
#         num = 2 * outputs * labels
#         den = outputs + labels
#         res = 1 - coef * ((num + 1)/(den + 1)).sum()
#         return res

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        num = 2 * (outputs * labels).sum()
        den = (outputs + labels).sum()
        res = 1 - (num) / (den + 1e-7)
        return res
    
    
class IoUloss(nn.Module):
    __name__= "IoU_Loss"

    """
    Implementation of IoU loss

    """
    def __init__(self, smooth = 1e-7):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        TP = (labels * outputs).sum() #TP
        FP = ((1 - labels) * outputs).sum() # FP
        FN = (labels * (1 - outputs)).sum() # FN
        IoU = TP / (TP + FP + FN + self.smooth)
        return 1 - IoU
    

    
class TverskyLoss(nn.Module):
    __name__= "Tversky_Loss"

    """
    Implementation of Tversky loss for image segmentation task.
    paper: https://arxiv.org/pdf/1706.05721.pdf

    """
    def __init__(self, alpha:float = 0.3, beta:float = 0.70):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        pp = (labels * outputs).sum()
        den1 = self.alpha * ((1 - labels) * outputs).sum() # FP
        den2 = self.beta * (labels * (1 - outputs)).sum() # FN
        tl = 1 - (1 + pp) / (1 + pp + den1 + den2 + 1e-7)
        return tl
    
class FocalTverskyLoss(nn.Module):
    __name__= "FocalTversky_Loss"

    """
    Implementation of Tversky loss for image segmentation task.
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py

    """
    def __init__(self, alpha:float = 0.3, beta:float = 0.7, gamma:float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        pp = (labels * outputs).sum()
        den1 = self.alpha * ((1 - labels) * outputs).sum() # FP
        den2 = self.beta * (labels * (1 - outputs)).sum() # FN
        tv_index = (1 + pp) / (1 + pp + den1 + den2 + 1e-7)
        ftl = torch.pow((1 - tv_index),self.gamma)
        return ftl
    
class FocalTverskyLoss_finetune(nn.Module):
    __name__= "FocalTversky_Loss_finetune"

    """
    Implementation of Tversky loss for image segmentation task.
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py

    """
    def __init__(self, alpha:float = 0.3, beta:float = 0.7, gamma:float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        pp = (labels * outputs).sum()
        den1 = self.alpha * ((1 - labels) * outputs).sum() # FP
        den2 = self.beta * (labels * (1 - outputs)).sum() # FN
        tv_index = (1 + pp) / (1 + pp + den1 + den2 + 1e-7)
        ftl = torch.pow((1 - tv_index),self.gamma)
        return ftl
                
class Unified_Focal_loss(nn.Module):
    __name__= "Unified_Focal"
                
    def __init__(self, alpha:float = 0.3, beta:float = 0.7, gamma:float = 0.25, delta:float = 0.75, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        if reduction not in ('mean', 'sum'):
            raise ValueError('"{}" is not a valid mode for reduction. Only "mean"'
                             'and "sum" are allowed.'.format(rediction))
        self.reduction = reduction
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs = torch.clamp(outputs, 0, 1)
        outputs = F.logsigmoid(outputs).exp()
        eps = 1e-6
        outputs = torch.clamp(outputs, min=eps, max=1.0 - eps)
        
        p_tb = (1 - outputs) * (1 - labels)
                
        focal_m = self.delta * labels * torch.log(outputs) + (1 - self.delta)*(torch.pow((1 - p_tb),self.gamma))*((1 - labels) * torch.log(1 - outputs))
        if self.reduction == 'mean':
            focal_m = -torch.mean(focal_m)
        elif self.reduction == 'sum':
            focal_m = -torch.sum(focal_m)
            
        TP = (labels * outputs).sum()
        FP = ((1 - labels) * outputs).sum() # FP
        FN = (labels * (1 - outputs)).sum() # FN
        tv_index = (TP + 1e-7) / (TP + self.alpha*FP + self.beta*FN + 1e-7)
        ftl = torch.pow((1 - tv_index),(1-self.gamma))
        
        
            
        unified_focal = 0.5*ftl + 0.5*focal_m
        
        return unified_focal
    
class Unified_Adaptive_Focal_loss(nn.Module):
    __name__= "Unified_Adaptive_Focal"
                
    def __init__(self, gamma:float = 0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        if reduction not in ('mean', 'sum'):
            raise ValueError('"{}" is not a valid mode for reduction. Only "mean"'
                             'and "sum" are allowed.'.format(rediction))
        self.reduction = reduction
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs = torch.clamp(outputs, 0, 1)
        outputs = F.logsigmoid(outputs).exp()
        eps = 1e-6
        outputs = torch.clamp(outputs, min=eps, max=1.0 - eps)
        
        p_tb = (1 - outputs) * (1 - labels)
        
        pos_labels: Tensor = labels.eq(1).sum()
        neg_labels: Tensor = labels.eq(0).sum()
        if pos_labels == 0:
            alpha = 1
            beta = 1
        else:
            alpha = 1 / torch.log(neg_labels / pos_labels)
            beta = 1 - alpha
                
        focal_m = alpha * labels * torch.log(outputs) + beta*(torch.pow((1 - p_tb),self.gamma))*((1 - labels) * torch.log(1 - outputs))
        if self.reduction == 'mean':
            focal_m = -torch.mean(focal_m)
        elif self.reduction == 'sum':
            focal_m = -torch.sum(focal_m)
            
        
            
        TP = (labels * outputs).sum()
        FP = ((1 - labels) * outputs).sum() # FP
        FN = (labels * (1 - outputs)).sum() # FN
        tv_index = (TP + 1e-7) / (TP + alpha*FP + beta*FN + 1e-7)
        aftl = torch.pow((1 - tv_index),(1-self.gamma))
        
            
        unified_adaptive_focal = aftl + focal_m
        
        return unified_adaptive_focal
                

    
class FLoss(nn.Module):
    __name__= "Fmeasure_Loss"

    """
    Implementation of Tversky loss for image segmentation task.
    paper: https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_Optimizing_the_F-Measure_for_Threshold-Free_Salient_Object_Detection_ICCV_2019_paper.pdf
    author code: https://github.com/zeakey/iccv2019-fmeasure

    """
    def __init__(self, beta:float = 0.3):
        super().__init__()
        self.beta = beta
    
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = F.logsigmoid(outputs).exp()
        tp = (labels * outputs).sum()
        fp = ((1 - labels) * outputs).sum()
        fn = (labels * (1 - outputs)).sum()
        num = (1 + self.beta**2)*tp
        den = (self.beta**2)*(tp + fn) + (tp + fp)
        fmeasure = num / (den + 1e-7)
        floss = 1 - fmeasure

        return floss 
    
class SSLoss(nn.Module):

    __name__= "SS_Loss"

    """
    refer from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
    
    Implementation of Sensitivity-Specifity loss for image segmentation task.
    paper: http://www.rogertam.ca/Brosch_MICCAI_2015.pdf
    tf code: https://github.com/NifTK/NiftyNet/blob/df0f86733357fdc92bbc191c8fec0dcf49aa5499/niftynet/layer/loss_segmentation.py

    """

    def __init__(
        self,
        batch_dice = False,
        from_logits=True,
        do_bg = True,
        square = False,
        #log_loss=False,
        smooth: float = 1e-7,
        ignore_index=None,
        #alpha: float = 0.3,
        r: float = 0.1,
        #gamma: float = 0.75
    ):
        """
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param batch_dice: sum tp, fp, fn in every batch
        :param do_bg: inclue background in loss computation r
        :param r: weight parameter in SS paper
        """
        super(SSLoss, self).__init__()

        self.batch_dice = batch_dice
        self.from_logits = from_logits
        self.do_bg = do_bg
        self.square = square
        # self.log_loss = log_loss
        self.smooth = smooth
        self.ignore_index = ignore_index
        # self.alpha = alpha
        self.r = r
        # self.gamma = gamma
        

    def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
        '''
        pred:   shape for binary segmentation                   - B x 1 x H x W
                shape for multiclass/multilable segmentation    - B x C x H x W
        gt:     shape for binary segmentation                   - B x 1 x H x W
                shape for multiclass segmentation               - B x H x W
                shape for multilable segmentation               - B x C x H x W (one-hot encoding)
        '''

        assert gt.size(0) == pred.size(0)

        pred_shape = pred.shape
        gt_shape = gt.shape

        if len(pred_shape) != len(gt_shape):
            gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
            gt_shape = gt.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(pred_shape)))
        else:
            axes = list(range(2, len(pred_shape)))

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if all([i == j for i, j in zip(pred_shape, gt_shape)]):
                pred = F.logsigmoid(pred).exp()
                # print('sigmoid:',pred)    
            else:
                pred = pred.log_softmax(dim=1).exp()
                # print('softmax', pred)

        if self.ignore_index is not None:
            mask = gt != self.ignore_index
            if all([i == j for i, j in zip(pred_shape, gt_shape)]):
                pred = pred * mask
                gt = gt * mask  
            else:
                pred = pred * mask.unsqueeze(1)
                gt = gt * mask
        
        with torch.no_grad():
            # if this is the case then gt is probably already a one hot encoding
            if all([i == j for i, j in zip(pred.shape, gt.shape)]):
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(pred_shape)
                if pred.device.type == "cuda":
                    y_onehot = y_onehot.cuda(pred.device.index)
                y_onehot.scatter_(1, gt, 1)


        # no object value
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - pred)**2
        specificity_part = torch.sum(squared_error*y_onehot, axes)/(torch.sum(y_onehot, axes)+self.smooth)
        sensitivity_part = torch.sum(squared_error*bg_onehot, axes)/(torch.sum(bg_onehot, axes)+self.smooth)

        loss = self.r * specificity_part + (1-self.r) * sensitivity_part

        if not self.do_bg:
            if self.batch_dice:
                loss = loss[1:]
            else:
                loss = loss[:, 1:]

        return loss.mean()    

# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/67791
class LovaszLoss(nn.Module):
    __name__= "Lovasz_Loss"
    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs = torch.sigmoid(outputs)
        outputs = outputs.flatten()
        labels = labels.flatten()
        signs = 2 * labels.float() - 1
        errors = (1 - outputs * Variable(signs))
        errors_sorted, indices = torch.sort(errors, dim=0, descending=True)
        gt_sorted = labels[indices.data]

        # gradient
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        grad = 1. - intersection / union

        p = len(gt_sorted)
        if p > 1: # cover 1-pixel case
            grad[1:p] = grad[1:p] - grad[0:-1]
       
        loss = torch.dot(torch.relu(errors_sorted), Variable(grad))
        return loss
    
#PyTorch
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook

# def flatten_binary_scores(scores, labels, ignore=None):
#     """
#     Flattens predictions in the batch (binary case)
#     Remove labels equal to 'ignore'
#     """
#     scores = scores.view(-1)
#     labels = labels.view(-1)
#     if ignore is None:
#         return scores, labels
#     valid = (labels != ignore)
#     vscores = scores[valid]
#     vlabels = labels[valid]
#     return vscores, vlabels

# def lovasz_grad(gt_sorted):
#     """
#     Computes gradient of the Lovasz extension w.r.t sorted errors
#     See Alg. 1 in paper
#     """
#     p = len(gt_sorted)
#     gts = gt_sorted.sum()
#     intersection = gts - gt_sorted.float().cumsum(0)
#     union = gts + (1 - gt_sorted).float().cumsum(0)
#     jaccard = 1. - intersection / union
#     if p > 1: # cover 1-pixel case
#         jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
#     return jaccard

# def lovasz_hinge(logits, labels, per_image=True, ignore=None):
#     """
#     Binary Lovasz hinge loss
#       logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
#       labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
#       per_image: compute the loss per image instead of per batch
#       ignore: void class id
#     """
#     if per_image:
#         loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
#                           for log, lab in zip(logits, labels))
#     else:
#         loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
#     return loss

# def lovasz_hinge_flat(logits, labels):
#     """
#     Binary Lovasz hinge loss
#       logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
#       labels: [P] Tensor, binary ground truth labels (0 or 1)
#       ignore: label to ignore
#     """
#     if len(labels) == 0:
#         # only void pixels, the gradients should be 0
#         return logits.sum() * 0.
#     signs = 2. * labels.float() - 1.
#     errors = (1. - logits * Variable(signs))
#     errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
#     perm = perm.data
#     gt_sorted = labels[perm]
#     grad = lovasz_grad(gt_sorted)
#     loss = torch.dot(F.relu(errors_sorted), Variable(grad))
#     return loss

# #=====
# #Multi-class Lovasz loss
# #=====

# def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
#     """
#     Multi-class Lovasz-Softmax loss
#       probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
#               Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
#       labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
#       classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#       per_image: compute the loss per image instead of per batch
#       ignore: void class labels
#     """
#     if per_image:
#         loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
#                           for prob, lab in zip(probas, labels))
#     else:
#         loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
#     return loss


# def lovasz_softmax_flat(probas, labels, classes='present'):
#     """
#     Multi-class Lovasz-Softmax loss
#       probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
#       labels: [P] Tensor, ground truth labels (between 0 and C - 1)
#       classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#     """
#     if probas.numel() == 0:
#         # only void pixels, the gradients should be 0
#         return probas * 0.
#     C = probas.size(1)
#     losses = []
#     class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
#     for c in class_to_sum:
#         fg = (labels == c).float() # foreground for class c
#         if (classes is 'present' and fg.sum() == 0):
#             continue
#         if C == 1:
#             if len(classes) > 1:
#                 raise ValueError('Sigmoid output possible only with 1 class')
#             class_pred = probas[:, 0]
#         else:
#             class_pred = probas[:, c]
#         errors = (Variable(fg) - class_pred).abs()
#         errors_sorted, perm = torch.sort(errors, 0, descending=True)
#         perm = perm.data
#         fg_sorted = fg[perm]
#         losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
#     return mean(losses)    

# #PyTorch
# class LovaszLoss(nn.Module):
#     __name__= "Lovasz_Loss"
#     def __init__(self, weight=None, size_average=True):
#         super(LovaszLoss, self).__init__()

#     def forward(self, inputs, targets):
#         inputs = F.sigmoid(inputs)    
#         Lovasz = lovasz_hinge(inputs, targets, per_image=False)                       
#         return Lovasz

    
class Dice_BCE(nn.Module):
    __name__= "Dice_BCE"
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError('"{}" is not a valid mode for reduction. Only "mean"'
                             'and "sum" are allowed.'.format(rediction))
        self.reduction = reduction
        
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        # outputs = torch.clamp(outputs, 0, 1)
        outputs = F.logsigmoid(outputs).exp()
        eps = 1e-7
        outputs = torch.clamp(outputs, min=eps, max=1.0 - eps)
        bce = labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs)
        num = 2 * (outputs * labels).sum()
        den = (outputs + labels).sum()
        res = 1 - (num) / (den + 1e-7)
        if self.reduction == 'mean':
            bce = -torch.mean(bce)
        elif self.reduction == 'sum':
            bce = -torch.sum(bce)
            
        dice_bce = res + bce
        
        return dice_bce
    
    
class Dice_Focal(_Loss):
    
    __name__= "Dice_Focal"


    def __init__(
        self,
        alpha=0.25,
        gamma: float = 2.0,
        beta = 1.0,
        ignore_index=None,
        reduction="mean",
        normalized=False,
        reduced_threshold=None,
    ):

        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
            ignore_index=ignore_index,
        )

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):

        focal = self.focal_loss_fn(outputs, labels)
        outputs = F.logsigmoid(outputs).exp()
        num = 2 * (outputs * labels).sum()
        den = (outputs + labels).sum()
        res = 1 - (num) / (den + 1e-7)
        dice_focal = res + self.beta*focal
        return dice_focal
    
# class Dice_Focal(nn.Module):
#     __name__= "Dice_Focal"
    
#     def __init__(self, alpha: int = 0.25, gamma: int = 2):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
        
#     def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
#         bce_logit = nn.BCEWithLogitsLoss()
#         ce = bce_logit(outputs, labels)
#         pt = torch.exp(-ce)
#         fl = self.alpha * torch.pow((1 - pt), self.gamma) * ce
#         num = 2 * (outputs * labels).sum()
#         den = (outputs + labels).sum()
#         dice_loss = 1 - (num) / (den + 1e-7)
#         log_cosh_dice_loss = torch.log((torch.exp(dice_loss) + torch.exp(-dice_loss))/2)
#         dice_focal = log_cosh_dice_loss + fl
#         return dice_focal
        
###******************************************************************************************************************************
###******************************************************************************************************************************
###******************************************************************************************************************************


#-----------------------------------------------------------------------------------------
# class SoftBCEWithLogitsLoss(_Loss):
#     __name__ = "Soft_BCEWithLogitsLoss"
#     '''
#     BCEWithLogitsLoss is from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCEWithLogitsLoss

#     This loss combines a `Sigmoid` layer and the `BCELoss` in one single class. 
#     This version is more numerically stable than using a plain `Sigmoid`
#     followed by a `BCELoss` as combining the operations into one layer
#     takes advantage of the log-sum-exp trick for numerical stability.
    
#     - input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
#     - targer: Tensor of the same shape as input with values between 0 and 1
#     - reduction:'none': no reduction will be applied, 
#                 'mean': the sum of the output will be divided by the number of elements in the output,
#                 'sum': the output will be summed (will be deprecated in the feature)
#     - ignore_index: If not None, targets may contain values to be ignored.
#                     Target values equal to ignore_index will be ignored from loss computation.
#     - smooth_factor: Label Smoothing prevents the network from becoming over-confident
#     - output: scalar. If reduction is 'none', the output has the same shape as input

#     - Support of ignore_index value
#     - Support of label smoothing
    
#     BCEWithLogitsLoss is from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/soft_bce.py
#     '''
#     __constants__ = ["weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"]

#     def __init__(self, 
#                  weight: Optional[Tensor] = None, 
#                 #  size_average=None, 
#                 #  reduce=None, 
#                  reduction: str = 'mean',
#                  ignore_index: Optional[int] = -100,
#                  smooth_factor = None,
#                  pos_weight=None):

#         super().__init__()
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth_factor = smooth_factor
#         self.register_buffer("weight", weight)
#         self.register_buffer("pos_weight", pos_weight)

#     def forward(self, input: Tensor, target: Tensor)-> torch.Tensor:
#         """
#         Args:
#             y_pred: torch.Tensor of shape (N, 1, H, W)
#             y_true: torch.Tensor of shape (N, 1, H, W)

#         Returns:
#             loss: torch.Tensor
#         """


#         if self.smooth_factor is not None:
#             soft_targets = ((1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)).type_as(input)
#         else:
#             soft_targets = target.type_as(input)

#         loss = F.binary_cross_entropy_with_logits(input, soft_targets, weight=self.weight, pos_weight=self.pos_weight, reduction="none")

#         if self.ignore_index is not None:
#             not_ignored_mask: Tensor = target != self.ignore_index
#             loss *= not_ignored_mask.type_as(loss)

#         if self.reduction == "mean": # Default to backward calculate loss (return scalar)
#             return loss.mean()

#         if self.reduction == "sum":
#             return loss.sum()

#         return loss
    

# class WBCEWithLogitLoss(nn.Module):

#     __name__ = "WBCEWithLogitLoss"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, weight=0.25, ignore_index=None, reduction='mean', smooth = None):
#         super(WBCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         weight = float(weight)
#         self.weight = weight # how to set weight
#         self.reduction = reduction
#         self.smooth = smooth

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         batch_size = output.size(0)
#         output = output.view(batch_size, -1)
#         target = target.view(batch_size, -1)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

#         # loss = self.bce(output, target)
#         loss = -self.weight * target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
# class mBCEWithLogitLoss(nn.Module):

#     __name__ = "mBCEWithLogitLoss"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Binary Cross Entropy.
#     `BCE(p,t)=-*t*log(p)-(1-t)*log(1-p)`
#     Args:
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, ignore_index=None, reduction='mean', smooth = None):
#         super(mBCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth = smooth

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         batch_size = output.size(0)
#         output = output.view(batch_size, -1)
#         target = target.view(batch_size, -1)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

#         # loss = self.bce(output, target)
#         loss = -target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
# class mMFW_BCEWithLogitLoss(nn.Module):

#     __name__ = "mMFW_BCEWithLogitLoss"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, pos_weight=1.0, neg_weight=1.0, ignore_index=None, reduction='mean', smooth = None):
#         super(mMFW_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         pos_weight = float(pos_weight)
#         neg_weight = float(neg_weight)
#         self.pos_weight = pos_weight # how to set weight
#         self.neg_weight = neg_weight
#         self.reduction = reduction
#         self.smooth = smooth

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         batch_size = output.size(0)
#         output = output.view(batch_size, -1)
#         target = target.view(batch_size, -1)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

#         # loss = self.bce(output, target)
#         loss = -self.pos_weight * target.mul(torch.log(output)) - self.neg_weight*((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
# class mIFW_BCEWithLogitLoss(nn.Module):

#     __name__ = "mIFW_BCEWithLogitLoss"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, pos_weight=1.0, neg_weight=1.0, ignore_index=None, reduction='mean', smooth = None):
#         super(mIFW_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         pos_weight = float(pos_weight)
#         neg_weight = float(neg_weight)
#         self.pos_weight = pos_weight # how to set weight
#         self.neg_weight = neg_weight
#         self.reduction = reduction
#         self.smooth = smooth

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         batch_size = output.size(0)
#         output = output.view(batch_size, -1)
#         target = target.view(batch_size, -1)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)

#         # loss = self.bce(output, target)
#         loss = -self.pos_weight * target.mul(torch.log(output)) - self.neg_weight*((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    

# class Adaptive_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "Adaptive_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, beta=0.75, ignore_index=None, reduction='mean', smooth = None):
#         super(Adaptive_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.beta = float(beta) # how to set weight
#         self.reduction = reduction
#         self.smooth = smooth

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         batch_size = output.size(0)
#         output = output.view(batch_size, -1)
#         target = target.view(batch_size, -1)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         pos_target: Tensor = target.eq(1).sum()
#         neg_target: Tensor = target.eq(0).sum()
#         num_target = pos_target + neg_target
#         alpha = neg_target / num_target
#         q_alpha = self.beta * (10**(2*alpha - 1))
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -q_alpha * target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss    

    
# def balanced_binary_cross_entropy_with_logits(
#     logits: Tensor, targets: Tensor, gamma: float = 1.0, ignore_index: Optional[int] = None, reduction: str = "mean"
# ) -> Tensor:
#     """
#     Balanced binary cross entropy loss.
#     Args:
#         logits:
#         targets: This loss function expects target values to be hard targets 0/1.
#         gamma: Power factor for balancing weights
#         ignore_index:
#         reduction:
#     Returns:
#         Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
#         shape of `logits` tensor.
#     """
#     pos_targets: Tensor = targets.eq(1).sum()
#     neg_targets: Tensor = targets.eq(0).sum()

#     num_targets = pos_targets + neg_targets
#     pos_weight = torch.pow(neg_targets / (num_targets + 1e-7), gamma)
#     neg_weight = 1.0 - pos_weight

#     pos_term = pos_weight.pow(gamma) * targets * torch.nn.functional.logsigmoid(logits)
#     neg_term = neg_weight.pow(gamma) * (1 - targets) * torch.nn.functional.logsigmoid(-logits)

#     loss = -(pos_term + neg_term)

#     if ignore_index is not None:
#         loss = torch.masked_fill(loss, targets.eq(ignore_index), 0)

#     if reduction == "mean":
#         loss = loss.mean()

#     if reduction == "sum":
#         loss = loss.sum()

#     return loss


# class BalancedBCEWithLogitsLoss(nn.Module):
#     __name__ = "BalancedBCEWithLogitsLoss"
#     """
#     Copy from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/balanced_bce.py
    
#     Balanced binary cross-entropy loss.
#     https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
#     """

#     __constants__ = ["gamma", "reduction", "ignore_index"]

#     def __init__(self, gamma: float = 1.0, reduction="mean", ignore_index: Optional[int] = None):
#         """
#         Args:
#             gamma:
#             ignore_index:
#             reduction:
#         """
#         super().__init__()
#         self.gamma = gamma
#         self.reduction = reduction
#         self.ignore_index = ignore_index

#     def forward(self, output: Tensor, target: Tensor) -> Tensor:
#         return balanced_binary_cross_entropy_with_logits(
#             output, target, gamma=self.gamma, ignore_index=self.ignore_index, reduction=self.reduction
#         )
    
# #-------------------------------------------------------------------------------------------------------------

# def IFW_binary_cross_entropy_with_logits(
#     logits: Tensor, targets: Tensor, gamma: float = 1.0, ignore_index: Optional[int] = None, reduction: str = "mean"
# ) -> Tensor:
#     """
#     Balanced binary cross entropy loss.
#     Args:
#         logits:
#         targets: This loss function expects target values to be hard targets 0/1.
#         gamma: Power factor for balancing weights
#         ignore_index:
#         reduction:
#     Returns:
#         Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
#         shape of `logits` tensor.
#     """
#     pos_targets: Tensor = targets.eq(1).sum()
#     neg_targets: Tensor = targets.eq(0).sum()

#     num_targets = pos_targets + neg_targets
#     if pos_targets == 0:
#         pos_weight = 1
#     else:
#         pos_weight = torch.pow(num_targets / pos_targets , gamma)
#     neg_weight = torch.pow(num_targets / neg_targets , gamma)

#     pos_term = pos_weight * targets * torch.nn.functional.logsigmoid(logits)
#     neg_term = neg_weight * (1 - targets) * torch.nn.functional.logsigmoid(-logits)

#     loss = -(pos_term + neg_term)

#     if ignore_index is not None:
#         loss = torch.masked_fill(loss, targets.eq(ignore_index), 0)

#     if reduction == "mean":
#         loss = loss.mean()

#     if reduction == "sum":
#         loss = loss.sum()

#     return loss


# class IFW_BCEWithLogitsLoss(nn.Module):
#     __name__ = "IFW_BCEWithLogitsLoss"
#     """
#     Copy from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/balanced_bce.py
    
#     Balanced binary cross-entropy loss.
#     https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
#     """

#     __constants__ = ["gamma", "reduction", "ignore_index"]

#     def __init__(self, gamma: float = 1.0, reduction="mean", ignore_index: Optional[int] = None):
#         """
#         Args:
#             gamma:
#             ignore_index:
#             reduction:
#         """
#         super().__init__()
#         self.gamma = gamma
#         self.reduction = reduction
#         self.ignore_index = ignore_index

#     def forward(self, output: Tensor, target: Tensor) -> Tensor:
#         return IFW_binary_cross_entropy_with_logits(
#             output, target, gamma=self.gamma, ignore_index=self.ignore_index, reduction=self.reduction
#         )

# #----------------------------------------------------------------------------------------------------------------

# def MFW_binary_cross_entropy_with_logits(
#     logits: Tensor, targets: Tensor, ignore_index: Optional[int] = None, reduction: str = "mean"
# ) -> Tensor:
#     """
#     Balanced binary cross entropy loss.
#     Args:
#         logits:
#         targets: This loss function expects target values to be hard targets 0/1.
#         gamma: Power factor for balancing weights
#         ignore_index:
#         reduction:
#     Returns:
#         Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
#         shape of `logits` tensor.
#     """
#     pos_targets: Tensor = targets.eq(1).sum()
#     neg_targets: Tensor = targets.eq(0).sum()
#     num_targets = pos_targets + neg_targets
#     if pos_targets == 0:
#         pos_weight = 1
#     else:
#         pos_weight = pos_targets / num_targets
#     neg_weight = neg_targets / num_targets
    
#     value = min(pos_weight, neg_weight) # median value for two classes
#     reverse_pos_weight = value / pos_weight
#     reverse_neg_weight = value / neg_weight

#     pos_term = pos_weight * targets * torch.nn.functional.logsigmoid(logits)
#     neg_term = neg_weight * (1 - targets) * torch.nn.functional.logsigmoid(-logits)

#     loss = -(pos_term + neg_term)

#     if ignore_index is not None:
#         loss = torch.masked_fill(loss, targets.eq(ignore_index), 0)

#     if reduction == "mean":
#         loss = loss.mean()

#     if reduction == "sum":
#         loss = loss.sum()

#     return loss


# class MFW_BCEWithLogitsLoss(nn.Module):
#     __name__ = "MFW_BCEWithLogitsLoss"
#     """
#     Copy from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/balanced_bce.py
    
#     Balanced binary cross-entropy loss.
#     https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
#     """

#     __constants__ = ["gamma", "reduction", "ignore_index"]

#     def __init__(self, reduction="mean", ignore_index: Optional[int] = None):
#         """
#         Args:
#             gamma:
#             ignore_index:
#             reduction:
#         """
#         super().__init__()
#         self.reduction = reduction
#         self.ignore_index = ignore_index

#     def forward(self, output: Tensor, target: Tensor) -> Tensor:
#         return MFW_binary_cross_entropy_with_logits(
#             output, target, ignore_index=self.ignore_index, reduction=self.reduction
#         )

# #----------------------------------------------------------------------------------------------------------------
    
# def compute_dtm(gt: Tensor) -> Tensor:
#     fg_dist = np.zeros(gt.shape)
#     bg_dist = np.zeros(gt.shape)
#     if gt.shape[1] == 1:
#         dis_id = 0
#     else:
#         dis_id = 1
#     for b in range(gt.shape[0]): # batch size
#         for c in range(dis_id, gt.shape[1]): # class_num
#             posmask = gt[b][c].cpu()
#             negmask = 1-posmask
#             pos_dis = edt(posmask)
#             neg_dis = edt(negmask)
#             norm_pos_dis = (pos_dis)/(np.max(pos_dis) + 1e-7)
#             norm_neg_dis = (neg_dis)/(np.max(neg_dis) + 1e-7)
#             norm_pos_dis[np.isnan(norm_pos_dis)]= 0.0
#             norm_neg_dis[np.isnan(norm_neg_dis)]= 0.0
#             fg_dist[b][c] = norm_pos_dis
#             bg_dist[b][c] = norm_neg_dis
#     return torch.from_numpy(fg_dist), torch.from_numpy(bg_dist)

# # def mcompute_dtm(gt: Tensor) -> Tensor:
# #     fg_dist = np.zeros(gt.shape)
# #     bg_dist = np.zeros(gt.shape)
# #     if gt.shape[1] == 1:
# #         dis_id = 0
# #     else:
# #         dis_id = 1
# #     for b in range(gt.shape[0]): # batch size
# #         for c in range(dis_id, gt.shape[1]): # class_num
# #             if gt[b][c].eq(1).sum() == 0:
# #                 pos_dis = np.zeros(gt[b][c].shape)
# #                 neg_dis = np.zeros(gt[b][c].shape)
# #             else:
# #                 posmask = gt[b][c].cpu()
# #                 negmask = 1-posmask
# #                 pos_dis = edt(posmask)
# #                 neg_dis = edt(negmask)
# #             fg_dist[b][c] = (pos_dis)/(np.max(pos_dis) + 1e-7)
# #             bg_dist[b][c] = (neg_dis)/(np.max(neg_dis) + 1e-7)
# #     return torch.from_numpy(fg_dist), torch.from_numpy(bg_dist)

    
# def dtm_binary_cross_entropy_with_logits(
#     logits: Tensor, targets: Tensor, ignore_index: Optional[int] = None, reduction: str = "mean", 
# device = 'cpu') -> Tensor:
#     """
#     Balanced binary cross entropy loss.
#     Args:
#         logits:
#         targets: This loss function expects target values to be hard targets 0/1.
#         gamma: Power factor for balancing weights
#         ignore_index:
#         reduction:
#     Returns:
#         Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
#         shape of `logits` tensor.
#     """
# #     pos_targets: Tensor = targets.eq(1).sum()
# #     neg_targets: Tensor = targets.eq(0).sum()

# #     num_targets = pos_targets + neg_targets
# #     pos_weight = torch.pow(neg_targets / (num_targets + 1e-7), gamma)
# #     neg_weight = 1.0 - pos_weight
#     preds = torch.sigmoid(logits)
#     fg_dist, bg_dist = compute_dtm(targets)
#     print('fg_dist:\n',np.unique(np.isnan(fg_dist.numpy())))
#     print('bg_dist:\n',np.unique(np.isnan(bg_dist.numpy())))
#     fg_dist = (1.0+fg_dist).to(device)
#     bg_dist = (1.0+bg_dist).to(device)
#     pos_term = fg_dist * targets * torch.log(preds)
#     # print('pos_term:\n',pos_term)
#     neg_term = bg_dist * (1 - targets) * torch.log(1-preds)
#     # print('neg_term:\n',neg_term)

#     loss = -(pos_term + neg_term)

#     if ignore_index is not None:
#         loss = torch.masked_fill(loss, targets.eq(ignore_index), 0)

#     if reduction == "mean":
#         loss = loss.mean()

#     if reduction == "sum":
#         loss = loss.sum()

#     return loss

# class DTM_BCEWithLogitsLoss(nn.Module):
#     __name__ = "DTM_BCEWithLogitsLoss"
#     """
#     Copy from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/balanced_bce.py
    
#     Balanced binary cross-entropy loss.
#     https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
#     """

#     __constants__ = ["gamma", "reduction", "ignore_index"]

#     def __init__(self,  reduction="mean", ignore_index: Optional[int] = None, device = 'cpu'):
#         """
#         Args:
#             gamma:
#             ignore_index:
#             reduction:
#         """
#         super().__init__()
#         # self.gamma = gamma
#         self.reduction = reduction
#         self.ignore_index = ignore_index
#         self.device = device

#     def forward(self, output: Tensor, target: Tensor) -> Tensor:
#         return dtm_binary_cross_entropy_with_logits(
#             output, target, ignore_index=self.ignore_index, reduction=self.reduction, device = self.device
#         )
    
# # def mdtm_binary_cross_entropy_with_logits(
# #     logits: Tensor, targets: Tensor, gamma: float = 1.0, ignore_index: Optional[int] = None, reduction: str = "mean", 
# # device = 'cpu') -> Tensor:
# #     """
# #     Balanced binary cross entropy loss.
# #     Args:
# #         logits:
# #         targets: This loss function expects target values to be hard targets 0/1.
# #         gamma: Power factor for balancing weights
# #         ignore_index:
# #         reduction:
# #     Returns:
# #         Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
# #         shape of `logits` tensor.
# #     """
# # #     pos_targets: Tensor = targets.eq(1).sum()
# # #     neg_targets: Tensor = targets.eq(0).sum()

# # #     num_targets = pos_targets + neg_targets
# # #     pos_weight = torch.pow(neg_targets / (num_targets + 1e-7), gamma)
# # #     neg_weight = 1.0 - pos_weight
# #     preds = torch.sigmoid(logits)
# #     fg_dist, bg_dist = mcompute_dtm(targets)
# #     fg_dist = (1+fg_dist).to(device)
# #     bg_dist = (1+bg_dist).to(device)
# #     pos_term = fg_dist.pow(gamma) * targets * torch.log(preds)
# #     neg_term = bg_dist.pow(gamma) * (1 - targets) * torch.log(1-preds)

# #     loss = -(pos_term + neg_term)

# #     if ignore_index is not None:
# #         loss = torch.masked_fill(loss, targets.eq(ignore_index), 0)

# #     if reduction == "mean":
# #         loss = loss.mean()

# #     if reduction == "sum":
# #         loss = loss.sum()

# #     return loss

# # class mDTM_BCEWithLogitsLoss(nn.Module):
# #     __name__ = "mDTM_BCEWithLogitsLoss"
# #     """
# #     Copy from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/balanced_bce.py
    
# #     Balanced binary cross-entropy loss.
# #     https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
# #     """

# #     __constants__ = ["gamma", "reduction", "ignore_index"]

# #     def __init__(self, gamma: float = 1.0, reduction="mean", ignore_index: Optional[int] = None, device = 'cpu'):
# #         """
# #         Args:
# #             gamma:
# #             ignore_index:
# #             reduction:
# #         """
# #         super().__init__()
# #         self.gamma = gamma
# #         self.reduction = reduction
# #         self.ignore_index = ignore_index
# #         self.device = device

# #     def forward(self, output: Tensor, target: Tensor) -> Tensor:
# #         return mdtm_binary_cross_entropy_with_logits(
# #             output, target, gamma=self.gamma, ignore_index=self.ignore_index, reduction=self.reduction, device = self.device
# #         )
    
    
# class DTM_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "DTM_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
#         super(DTM_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth = smooth
#         self.device = device

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         fg_dist, _ = compute_dtm(target)
#         fg_dist = (1+fg_dist).to(self.device)
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -fg_dist * target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
# class mDTM_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "mDTM_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
#         super(mDTM_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth = smooth
#         self.device = device

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         fg_dist, bg_dist = compute_dtm(target)
#         fg_dist = (1+fg_dist).to(self.device)
#         bg_dist = (1+bg_dist).to(self.device)
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -fg_dist * target.mul(torch.log(output)) - bg_dist * ((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
# #--------------------------------------------------------------------------------------------------------------    
# def compute_dpt(gt: Tensor) -> Tensor:
#     dpt = np.zeros(gt.shape)
#     if gt.shape[1] == 1:
#         dis_id = 0
#     else:
#         dis_id = 1
#     for b in range(gt.shape[0]): # batch size
#         for c in range(dis_id, gt.shape[1]): # class_num
#             posmask = gt[b][c].cpu()
#             negmask = 1-posmask
#             pos_dis = edt(posmask)
#             neg_dis = edt(negmask)
#             dis = pos_dis + neg_dis
#             norm_dis = (dis)/(np.max(dis) + 1e-7)
#             inverse_dis = 1 - norm_dis
            
#             dpt[b][c] = inverse_dis
#     return torch.from_numpy(dpt)

# class DPT_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "DPT_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
#         super(DPT_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth = smooth
#         self.device = device

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         dpt = compute_dpt(target)
#         dpt = (1+dpt).to(self.device)
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -target.mul(torch.log(output)) - dpt*((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
# def __init__(self, pos_weight=1.0, neg_weight=1.0, ignore_index=None, reduction='mean', smooth = None):
#         super(mMFW_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         pos_weight = float(pos_weight)
#         neg_weight = float(neg_weight)
#         self.pos_weight = pos_weight # how to set weight
#         self.neg_weight = neg_weight
#         self.reduction = reduction
#         self.smooth = smooth
        
# class MFW_DPT_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "MFW_DPT_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, pos_weight=1.0, neg_weight=1.0, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
#         super(MFW_DPT_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         pos_weight = float(pos_weight)
#         neg_weight = float(neg_weight)
#         self.pos_weight = pos_weight # how to set weight
#         self.neg_weight = neg_weight
#         self.reduction = reduction
#         self.smooth = smooth
#         self.device = device

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         dpt = compute_dpt(target)
#         dpt = (1+dpt).to(self.device)
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -self.pos_weight*target.mul(torch.log(output)) - self.neg_weight*(dpt*((1.0 - target).mul(torch.log(1.0 - output))))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
# class mDPT_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "mDPT_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
#         super(mDPT_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth = smooth
#         self.device = device

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         dpt = compute_dpt(target)
#         dpt = (1+dpt).to(self.device)
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -dpt*target.mul(torch.log(output)) - ((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
# class pDPT_nDPT_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "pDPT_nDPT_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
#         super(pDPT_nDPT_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth = smooth
#         self.device = device

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         dpt = compute_dpt(target)
#         dpt = (1+dpt).to(self.device)
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -dpt * target.mul(torch.log(output)) - dpt * ((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
# def compute_pdtm_ndpt(gt: Tensor) -> Tensor:
#     fg_dist = np.zeros(gt.shape)
#     dpt = np.zeros(gt.shape)
#     if gt.shape[1] == 1:
#         dis_id = 0
#     else:
#         dis_id = 1
#     for b in range(gt.shape[0]): # batch size
#         for c in range(dis_id, gt.shape[1]): # class_num
#             posmask = gt[b][c].cpu()
#             negmask = 1-posmask
#             pos_dis = edt(posmask)
#             neg_dis = edt(negmask)
#             dis = pos_dis + neg_dis
#             norm_dis = (dis)/(np.max(dis) + 1e-7)
#             norm_dis[np.isnan(norm_dis)]= 0.0
#             inverse_dis = 1 - norm_dis
#             norm_pos_dis = (pos_dis)/(np.max(pos_dis) + 1e-7)
#             norm_pos_dis[np.isnan(norm_pos_dis)]= 0.0
            
#             fg_dist[b][c] = norm_pos_dis
#             dpt[b][c] = inverse_dis
#     return torch.from_numpy(fg_dist), torch.from_numpy(dpt)



# class pDTM_nDPT_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "pDTM_nDPT_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
#         super(pDTM_nDPT_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth = smooth
#         self.device = device

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         fg_dist, dpt = compute_pdtm_ndpt(target)
#         fg_dist = (1+fg_dist).to(self.device)
#         dpt = (1+dpt).to(self.device)
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -fg_dist*target.mul(torch.log(output)) - dpt*((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
    
# def compute_ndtm_pdpt(gt: Tensor) -> Tensor:
#     bg_dist = np.zeros(gt.shape)
#     dpt = np.zeros(gt.shape)
#     if gt.shape[1] == 1:
#         dis_id = 0
#     else:
#         dis_id = 1
#     for b in range(gt.shape[0]): # batch size
#         for c in range(dis_id, gt.shape[1]): # class_num
#             posmask = gt[b][c].cpu()
#             negmask = 1-posmask
#             pos_dis = edt(posmask)
#             neg_dis = edt(negmask)
#             dis = pos_dis + neg_dis
#             norm_dis = (dis)/(np.max(dis) + 1e-7)
#             norm_dis[np.isnan(norm_dis)]= 0.0
#             inverse_dis = 1 - norm_dis
#             norm_neg_dis = (neg_dis)/(np.max(neg_dis) + 1e-7)
#             norm_neg_dis[np.isnan(norm_neg_dis)]= 0.0
            
#             bg_dist[b][c] = norm_neg_dis
#             dpt[b][c] = inverse_dis
#     return torch.from_numpy(bg_dist), torch.from_numpy(dpt)

    
# class nDTM_pDPT_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "nDTM_pDPT_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu'):
#         super(nDTM_pDPT_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth = smooth
#         self.device = device

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         bg_dist, dpt = compute_ndtm_pdpt(target)
#         bg_dist = (1+bg_dist).to(self.device)
#         dpt = (1+dpt).to(self.device)
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -dpt*target.mul(torch.log(output)) - bg_dist*((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
     

# class pAW_nDPT_Weighted_BCEWithLogitLoss(nn.Module):

#     __name__ = "pAW_nDPT_Weighted_BCE"

#     """
#     Reference: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss
    
#     Adaptive_Weighted Binary Cross Entropy.
#     `WBCE(p,t)=-q_alpha*t*log(p)-(1-t)*log(1-p)`
#     To decrease the number of false negatives, set β>1. 
#     To decrease the number of false positives, set β<1. 
#     Args:
#             @param weight: positive sample weight
#         Shapes：
#             output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
#             target: A tensor of shape same with output
#     """

#     def __init__(self, ignore_index=None, reduction='mean', smooth = None, device = 'cpu', beta = 0.20):
#         super(pAW_nDPT_Weighted_BCEWithLogitLoss, self).__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.smooth = smooth
#         self.device = device
#         self.beta = beta

#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"

#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)

#         output = torch.sigmoid(output) # should be replace with logsigmoid function
#         # print(output)
#         # avoid `nan` loss
#         eps = 1e-6
#         output = torch.clamp(output, min=eps, max=1.0 - eps)
#         # soft label
#         if self.smooth is not None:
#             target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)
            
#         pos_target: Tensor = target.eq(1).sum()
#         neg_target: Tensor = target.eq(0).sum()
#         num_target = pos_target + neg_target
#         alpha = neg_target / num_target
#         q_alpha = self.beta * (10**(2*alpha - 1))
        
#         dpt = compute_dpt(target)
#         dpt = (1+dpt).to(self.device)
#         # print(pos_target, neg_target, num_target, alpha, q_alpha)

#         # loss = self.bce(output, target)
#         loss = -q_alpha*target.mul(torch.log(output)) - dpt*((1.0 - target).mul(torch.log(1.0 - output)))
#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             loss = loss
#         else:
#             raise NotImplementedError
#         return loss
    
    
# def focal_loss_with_logits(
#     output: torch.Tensor,
#     target: torch.Tensor,
#     gamma: float = 2.0,
#     alpha: Optional[float] = 0.25,
#     reduction: str = "mean",
#     normalized: bool = False,
#     reduced_threshold: Optional[float] = None,
#     eps: float = 1e-6,
#     ignore_index=None,
# ) -> torch.Tensor:
#     """Compute binary focal loss between target and output logits.
#     See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
#     Args:
#         output: Tensor of arbitrary shape (predictions of the model)
#         target: Tensor of the same shape as input
#         gamma: Focal loss power factor
#         alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
#             high values will give more weight to positive class.
#         reduction (string, optional): Specifies the reduction to apply to the output:
#             'none': no reduction will be applied,
#             'mean': the sum of the output will be divided by the number of
#             elements in the output, 'sum': the output will be summed.
#             'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
#         normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
#         reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
#     References:
#         https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
#     """
#     target = target.type_as(output)

#     p = torch.sigmoid(output)
#     ce_loss = F.binary_cross_entropy_with_logits(output, target, reduction="none")
#     p_t = p * target + (1 - p) * (1 - target)

#     # compute the loss
#     if reduced_threshold is None:
#         focal_term = (1.0 - p_t).pow(gamma)
#     else:
#         focal_term = ((1.0 - p_t) / reduced_threshold).pow(gamma)
#         focal_term = torch.masked_fill(focal_term, p_t < reduced_threshold, 1)

#     loss = focal_term * ce_loss

#     if alpha is not None:
#         alpha_t = alpha * target + (1 - alpha) * (1 - target)
#         loss *= alpha_t

#     if ignore_index is not None:
#         ignore_mask = target.eq(ignore_index)
#         loss = torch.masked_fill(loss, ignore_mask, 0)
#         if normalized:
#             focal_term = torch.masked_fill(focal_term, ignore_mask, 0)

#     if normalized:
#         norm_factor = focal_term.sum(dtype=torch.float32).clamp_min(eps)
#         loss /= norm_factor

#     if reduction == "mean":
#         loss = loss.mean()
#     if reduction == "sum":
#         loss = loss.sum(dtype=torch.float32)
#     if reduction == "batchwise_mean":
#         loss = loss.sum(dim=0, dtype=torch.float32)

#     return loss


# class BinaryFocalLoss(_Loss):
    
#     __name__= "BinaryFocalLoss"


#     def __init__(
#         self,
#         alpha=None,
#         gamma: float = 2.0,
#         ignore_index=None,
#         reduction="mean",
#         normalized=False,
#         reduced_threshold=None,
#     ):
#         """
#         :param alpha: Prior probability of having positive value in target.
#         :param gamma: Power factor for dampening weight (focal strenght).
#         :param ignore_index: If not None, targets may contain values to be ignored.
#         Target values equal to ignore_index will be ignored from loss computation.
#         :param reduced: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
#         :param threshold:
#         """
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.focal_loss_fn = partial(
#             focal_loss_with_logits,
#             alpha=alpha,
#             gamma=gamma,
#             reduced_threshold=reduced_threshold,
#             reduction=reduction,
#             normalized=normalized,
#             ignore_index=ignore_index,
#         )

#     def forward(self, label_input, label_target):
#         """Compute focal loss for binary classification problem.
#         label_input shape and label target must be the same
#         """
#         loss = self.focal_loss_fn(label_input, label_target)
#         return loss
    
# class FocalLoss_b(_Loss):
    
#     __name__= "FocalLoss_bloss"


#     def __init__(
#         self,
#         alpha=0.25,
#         gamma: float = 2.0,
#         ignore_index=None,
#         reduction="mean",
#         normalized=False,
#         reduced_threshold=None,
#     ):
#         """
#         :param alpha: Prior probability of having positive value in target.
#         :param gamma: Power factor for dampening weight (focal strenght).
#         :param ignore_index: If not None, targets may contain values to be ignored.
#         Target values equal to ignore_index will be ignored from loss computation.
#         :param reduced: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
#         :param threshold:
#         """
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.focal_loss_fn = partial(
#             focal_loss_with_logits,
#             alpha=alpha,
#             gamma=gamma,
#             reduced_threshold=reduced_threshold,
#             reduction=reduction,
#             normalized=normalized,
#             ignore_index=ignore_index,
#         )

#     def forward(self, label_input, label_target):
#         """Compute focal loss for binary classification problem.
#         label_input shape and label target must be the same
#         """
#         loss = self.focal_loss_fn(label_input, label_target)
#         return loss
    
# #-------------------------------------------------Region-based_losses---------------------------------------
# ## Some helper functions
# def sum_tensor(inp, axes, keepdim=False):
#     # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
#     axes = np.unique(axes).astype(int)
#     if keepdim:
#         for ax in axes:
#             inp = inp.sum(int(ax), keepdim=True)
#     else:
#         for ax in sorted(axes, reverse=True):
#             inp = inp.sum(int(ax))
#     return inp

# # refer from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
# import torch
# from torch import Tensor
# from torch import nn
# from torch.autograd import Variable
# from torch import einsum
# import numpy as np

# def get_tp_fp_fn(pred, gt, axes=None, mask=None, square=False):
#     """
#     pred must be (B, C, H, W))
#     gt must be a label map (shape (B, 1, H, W) OR shape (B, H, W)) or one hot encoding (B, C, H, W)
#     if mask is provided it must have shape (B, 1, H, W))
#     :param pred:
#     :param gt:
#     :param axes:
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(pred.size())))

#     pred_shape = pred.shape
#     gt_shape = gt.shape

#     with torch.no_grad():
#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))

#         # if this is the case then gt is probably already a one hot encoding
#         if all([i == j for i, j in zip(pred.shape, gt.shape)]):
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(pred_shape)
#             if pred.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(pred.device.index)
#             y_onehot.scatter_(1, gt, 1)

#     tp = pred * y_onehot
#     fp = pred * (1 - y_onehot)
#     fn = (1 - pred) * y_onehot

#     if mask is not None:
#         tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
#         fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
#         fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

#     if square:
#         tp = tp ** 2
#         fp = fp ** 2
#         fn = fn ** 2

#     # tp = sum_tensor(tp, axes, keepdim=False)
#     # fp = sum_tensor(fp, axes, keepdim=False)
#     # fn = sum_tensor(fn, axes, keepdim=False)
#     tp = torch.sum(tp, axes)
#     fp = torch.sum(fp, axes)
#     fn = torch.sum(fn, axes)

#     return tp, fp, fn



# class DiceLoss(nn.Module):

#     __name__= "Dice_Loss"

#     """
#     Implementation of Dice loss for image segmentation task.

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         eps=1e-7,
#     ):
#         """
#         :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param eps: Small epsilon for numerical stability
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
        
#         paper: https://arxiv.org/pdf/1606.04797.pdf

#         """
#         super(DiceLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.eps = eps
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W or B x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         dice_score = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 dice_score = dice_score[1:]
#             else:
#                 dice_score = dice_score[:, 1:]

#         if self.log_loss:
#             loss = -torch.log(dice_score.clamp_min(self.eps))
#         else:
#             loss = 1.0 - dice_score

#         # Dice loss is undefined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss

#         return loss.mean()
    
# class IFW_DiceLoss(nn.Module):

#     __name__= "IFW_Dice_Loss"

#     """
#     Implementation of Dice loss for image segmentation task.

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         eps=1e-7,
#     ):
#         """
#         :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param eps: Small epsilon for numerical stability
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
        
#         paper: https://arxiv.org/pdf/1606.04797.pdf

#         """
#         super(IFW_DiceLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.eps = eps
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W or B x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask
            
#         pos_target = gt.eq(1).sum()
#         if pos_target == 0:
#             w = 1.0
#         else:
#             w = 1 / (pos_target**2)

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         ifw_dice_score = (w * (2 * tp + self.smooth)) / (w * (2 * tp + fp + fn + self.smooth))

#         if not self.do_bg:
#             if self.batch_dice:
#                 ifw_dice_score = ifw_dice_score[1:]
#             else:
#                 ifw_dice_score = ifw_dice_score[:, 1:]

#         if self.log_loss:
#             loss = -torch.log(ifw_dice_score.clamp_min(self.eps))
#         else:
#             loss = 1.0 - ifw_dice_score

#         # Dice loss is undefined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss

#         return loss.mean()
    
# class FocalW_DiceLoss(nn.Module):

#     __name__= "FocalW_Dice_Loss"

#     """
#     Implementation of Dice loss for image segmentation task.

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         eps=1e-7,
#     ):
#         """
#         :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param eps: Small epsilon for numerical stability
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
        
#         paper: https://arxiv.org/pdf/1606.04797.pdf

#         """
#         super(FocalW_DiceLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.eps = eps
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W or B x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask
            
#         # pos_target = gt.eq(1).sum()
#         # if pos_target == 0:
#         #     w = 1.0
#         # else:
#         #     w = 1 / (pos_target**2)
#         p_t = pred * gt + (1 - pred) * (1 - gt)

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         focalw_dice_score = ((1+p_t) * (2 * tp + self.smooth)) / ((1+p_t) * (2 * tp + fp + fn + self.smooth))

#         if not self.do_bg:
#             if self.batch_dice:
#                 focalw_dice_score = focalw_dice_score[1:]
#             else:
#                 focalw_dice_score = focalw_dice_score[:, 1:]

#         if self.log_loss:
#             loss = -torch.log(focalw_dice_score.clamp_min(self.eps))
#         else:
#             loss = 1.0 - focalw_dice_score

#         # Dice loss is undefined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss

#         return loss.mean()
    
# class DSC(nn.Module):

#     __name__= "DSC"

#     """
#     Implementation of Dice loss for image segmentation task.

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         # log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         eps=1e-7,
#     ):
#         """
#         :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param eps: Small epsilon for numerical stability
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
        
#         paper: https://arxiv.org/pdf/1606.04797.pdf

#         """
#         super(DSC, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.eps = eps
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W or B x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         dice_score = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 dice_score = dice_score[1:]
#             else:
#                 dice_score = dice_score[:, 1:]
                
#         DSC = -dice_score

#         # if self.log_loss:
#         #     loss = -torch.log(dice_score.clamp_min(self.eps))
#         # else:
#         #     loss = 1.0 - dice_score

#         # Dice loss is undefined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss

#         return DSC.mean()

# class JaccardLoss(nn.Module):

#     __name__= "Jaccard_Loss"

#     """
#     Implementation of Jaccard loss (IoU loss) for image segmentation task.

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         eps=1e-7,
#     ):
#         """
#         :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param eps: Small epsilon for numerical stability
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
        
#         paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22

#         """
#         super(JaccardLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.eps = eps
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         jaccard_score = (tp + self.smooth) / (tp + fp + fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 dice_score = jaccard_score[1:]
#             else:
#                 dice_score = jaccard_score[:, 1:]

#         if self.log_loss:
#             loss = -torch.log(jaccard_score.clamp_min(self.eps))
#         else:
#             loss = 1.0 - jaccard_score

#         # Dice loss is undefined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss

#         return loss.mean()
    
    
# class IFW_JaccardLoss(nn.Module):

#     __name__= "IFW_Jaccard_Loss"

#     """
#     Implementation of Jaccard loss (IoU loss) for image segmentation task.

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         eps=1e-7,
#     ):
#         """
#         :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param eps: Small epsilon for numerical stability
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
        
#         paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22

#         """
#         super(IFW_JaccardLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.eps = eps
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask
            
#         pos_target = gt.eq(1).sum()
#         if pos_target == 0:
#             w = 1.0
#         else:
#             w = 1 / (pos_target**2)

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         ifw_jaccard_score = (w * tp + self.smooth) / (w*(tp + fp + fn) + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 ifw_jaccard_score = ifw_jaccard_score[1:]
#             else:
#                 ifw_jaccard_score = ifw_jaccard_score[:, 1:]

#         if self.log_loss:
#             loss = -torch.log(ifw_jaccard_score.clamp_min(self.eps))
#         else:
#             loss = 1.0 - ifw_jaccard_score

#         # Dice loss is undefined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss

#         return loss.mean()


    
    
# class TverskyLoss(nn.Module):

#     __name__= "Tversky_Loss"

#     """
#     Implementation of Tversky loss for image segmentation task.
#     paper: https://arxiv.org/pdf/1706.05721.pdf

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         # log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         alpha: float = 0.3,
#         beta: float = 0.7
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
#         :param alpha: controls the penalty for false positives.
#         : param beta: penalty for false negative. Larger beta weigh recall higher
#         """
#         super(TverskyLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.alpha = alpha
#         self.beta = beta
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         tversky_index = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 tversky_index = tversky_index[1:]
#             else:
#                 tversky_index = tversky_index[:, 1:]

#         loss = 1.0 - tversky_index


#         return loss.mean()

# def dtp_get_tp_fp_fn(pred, gt, axes=None, device='cpu'):
#     """
#     pred must be (B, C, H, W))
#     gt must be a label map (shape (B, 1, H, W) OR shape (B, H, W)) or one hot encoding (B, C, H, W)
#     if mask is provided it must have shape (B, 1, H, W))
#     :param pred:
#     :param gt:
#     :param axes:
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(pred.size())))

#     pred_shape = pred.shape
#     gt_shape = gt.shape

#     with torch.no_grad():
#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))

#         # if this is the case then gt is probably already a one hot encoding
#         if all([i == j for i, j in zip(pred.shape, gt.shape)]):
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(pred_shape)
#             if pred.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(pred.device.index)
#             y_onehot.scatter_(1, gt, 1)

#     tp = pred * y_onehot
#     fp = pred * (1 - y_onehot)
#     fn = (1 - pred) * y_onehot
    
#     dpt = compute_dpt(y_onehot)
#     dpt = (1+dpt).to(device)
    
#     w_fp = dpt * fp
#     w_fn = dpt * fn

#     tp = torch.sum(tp, axes)
#     w_fp = torch.sum(w_fp, axes)
#     w_fn = torch.sum(w_fn, axes)

#     return tp, w_fp, w_fn
       
# class DPT_TverskyLoss(nn.Module):

#     __name__= "DPT_Tversky_Loss"

#     """
#     Implementation of Tversky loss for image segmentation task.
#     paper: https://arxiv.org/pdf/1706.05721.pdf

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         # log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         device = 'cpu'
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
#         :param alpha: controls the penalty for false positives.
#         : param beta: penalty for false negative. Larger beta weigh recall higher
#         """
#         super(DPT_TverskyLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.device = device


#     def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask

#         tp, w_fp, w_fn = dtp_get_tp_fp_fn(pred, gt, axes, self.device)
#         w_tversky_index = (tp + self.smooth) / (tp + w_fp + w_fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 w_tversky_index = w_tversky_index[1:]
#             else:
#                 w_tversky_index = w_tversky_index[:, 1:]

#         loss = 1.0 - w_tversky_index


#         return loss.mean()    
    
# def dtm_get_tp_fp_fn(pred, gt, axes=None, device='cpu'):
#     """
#     pred must be (B, C, H, W))
#     gt must be a label map (shape (B, 1, H, W) OR shape (B, H, W)) or one hot encoding (B, C, H, W)
#     if mask is provided it must have shape (B, 1, H, W))
#     :param pred:
#     :param gt:
#     :param axes:
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(pred.size())))

#     pred_shape = pred.shape
#     gt_shape = gt.shape

#     with torch.no_grad():
#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))

#         # if this is the case then gt is probably already a one hot encoding
#         if all([i == j for i, j in zip(pred.shape, gt.shape)]):
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(pred_shape)
#             if pred.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(pred.device.index)
#             y_onehot.scatter_(1, gt, 1)

#     tp = pred * y_onehot
#     fp = pred * (1 - y_onehot)
#     fn = (1 - pred) * y_onehot
    
#     dpt = compute_dpt(y_onehot)
#     dtm = 1 - dpt
#     dtm = (1+dtm).to(device)
    
#     w_fp = dtm * fp
#     w_fn = dtm * fn

#     tp = torch.sum(tp, axes)
#     w_fp = torch.sum(w_fp, axes)
#     w_fn = torch.sum(w_fn, axes)

#     return tp, w_fp, w_fn
       
# class DTM_TverskyLoss(nn.Module):

#     __name__= "DTM_TverskyLoss"

#     """
#     Implementation of Tversky loss for image segmentation task.
#     paper: https://arxiv.org/pdf/1706.05721.pdf

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         # log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         device = 'cpu'
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
#         :param alpha: controls the penalty for false positives.
#         : param beta: penalty for false negative. Larger beta weigh recall higher
#         """
#         super(DTM_TverskyLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.device = device


#     def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask

#         tp, w_fp, w_fn = dtm_get_tp_fp_fn(pred, gt, axes, self.device)
#         w_tversky_index = (tp + self.smooth) / (tp + w_fp + w_fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 w_tversky_index = w_tversky_index[1:]
#             else:
#                 w_tversky_index = w_tversky_index[:, 1:]

#         loss = 1.0 - w_tversky_index


#         return loss.mean()    
    
    
    
# class FocalTverskyLoss(nn.Module):

#     __name__= "FocalTversky_Loss"

#     """
#     Implementation of Tversky loss for image segmentation task.
#     paper: https://arxiv.org/pdf/1810.07842.pdf
#     author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         # log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         alpha: float = 0.3,
#         beta: float = 0.7,
#         gamma: float = 0.75
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
#         :param alpha: controls the penalty for false positives.
#         : param beta: penalty for false negative. Larger beta weigh recall higher
#         """
#         super(FocalTverskyLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         tversky_index = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 tversky_index = tversky_index[1:]
#             else:
#                 tversky_index = tversky_index[:, 1:]

#         tversky_loss = 1.0 - tversky_index
#         loss = torch.pow(tversky_loss, self.gamma)

#         return loss.mean()
    


# class AsymLoss(nn.Module):

#     __name__= "Asym_Loss"

#     """
#     refer from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
    
#     Implementation of Asymmetric similarity loss for image segmentation task.
#     This is a special case of Tversky loss when alpha + beta = 1
#     paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         #log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         #alpha: float = 0.3,
#         beta: float = 1.5,
#         #gamma: float = 0.75
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
#         :param alpha: controls the penalty for false positives.
#         :param beta: penalty for false negative. Larger beta weigh recall higher
#         :param gamma:
#         """
#         super(AsymLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         # self.alpha = alpha
#         self.beta = beta
#         # self.gamma = gamma
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = pred * mask
#                 gt = gt * mask  
#             else:
#                 pred = pred * mask.unsqueeze(1)
#                 gt = gt * mask

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         weight = (self.beta**2) / (1 + self.beta**2)
#         asym = (tp + self.smooth) / (tp + weight*fp + (1-weight)*fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 asym = asym[1:]
#             else:
#                 asym = asym[:, 1:]

#         loss = 1-asym

#         return loss.mean()
    
    
# class FLoss(nn.Module):

#     __name__= "F_Loss"

#     """
#     refer from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
    
#     Implementation of Asymmetric similarity loss for image segmentation task.
#     This is a special case of Tversky loss when alpha + beta = 1
#     paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         #log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         #alpha: float = 0.3,
#         beta: float = 0.3,
#         #gamma: float = 0.75
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
#         :param alpha: controls the penalty for false positives.
#         :param beta: penalty for false negative. Larger beta weigh recall higher
#         :param gamma:
#         """
#         super(FLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         # self.alpha = alpha
#         self.beta = beta
#         # self.gamma = gamma
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = pred * mask
#                 gt = gt * mask  
#             else:
#                 pred = pred * mask.unsqueeze(1)
#                 gt = gt * mask

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         # weight = (self.beta**2) / (1 + self.beta**2)
#         floss = (tp + self.smooth) / (tp + self.beta*fp + (1-self.beta)*fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 floss = floss[1:]
#             else:
#                 floss = floss[:, 1:]

#         loss = 1-floss

#         return loss.mean()    

# def dtp_get_tp_fp_fn_for_asym(pred, gt, axes=None, device='cpu'):
#     """
#     pred must be (B, C, H, W))
#     gt must be a label map (shape (B, 1, H, W) OR shape (B, H, W)) or one hot encoding (B, C, H, W)
#     if mask is provided it must have shape (B, 1, H, W))
#     :param pred:
#     :param gt:
#     :param axes:
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(pred.size())))

#     pred_shape = pred.shape
#     gt_shape = gt.shape

#     with torch.no_grad():
#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))

#         # if this is the case then gt is probably already a one hot encoding
#         if all([i == j for i, j in zip(pred.shape, gt.shape)]):
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(pred_shape)
#             if pred.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(pred.device.index)
#             y_onehot.scatter_(1, gt, 1)

#     tp = pred * y_onehot
#     fp = pred * (1 - y_onehot)
#     fn = (1 - pred) * y_onehot
    
#     dpt = compute_dpt(y_onehot)
#     dpt = (1+dpt).to(device)
#     weight = (dpt**2) / (1 + dpt**2)
    
#     w_fp = weight * fp
#     w_fn = (1-weight) * fn

#     tp = torch.sum(tp, axes)
#     w_fp = torch.sum(w_fp, axes)
#     w_fn = torch.sum(w_fn, axes)

#     return tp, w_fp, w_fn    
    
# class DPT_AsymLoss(nn.Module):

#     __name__= "DPT_Asym_Loss"

#     """
#     refer from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
    
#     Implementation of Asymmetric similarity loss for image segmentation task.
#     This is a special case of Tversky loss when alpha + beta = 1
#     paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         #log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         device = 'cpu'
#         #alpha: float = 0.3,
#         #gamma: float = 0.75
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
#         :param alpha: controls the penalty for false positives.
#         :param beta: penalty for false negative. Larger beta weigh recall higher
#         :param gamma:
#         """
#         super(DPT_AsymLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.device = device
#         # self.alpha = alpha
#         # self.gamma = gamma
        

#     def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = pred * mask
#                 gt = gt * mask  
#             else:
#                 pred = pred * mask.unsqueeze(1)
#                 gt = gt * mask

#         tp, w_fp, w_fn = dtp_get_tp_fp_fn_for_asym(pred, gt, axes, self.device)
#         # weight = (self.beta**2) / (1 + self.beta**2)
#         asym = (tp + self.smooth) / (tp + w_fp + w_fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 asym = asym[1:]
#             else:
#                 asym = asym[:, 1:]

#         loss = 1-asym

#         return loss.mean() 
    
    
# def dtm_get_tp_fp_fn_for_asym(pred, gt, axes=None, device='cpu'):
#     """
#     pred must be (B, C, H, W))
#     gt must be a label map (shape (B, 1, H, W) OR shape (B, H, W)) or one hot encoding (B, C, H, W)
#     if mask is provided it must have shape (B, 1, H, W))
#     :param pred:
#     :param gt:
#     :param axes:
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(pred.size())))

#     pred_shape = pred.shape
#     gt_shape = gt.shape

#     with torch.no_grad():
#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))

#         # if this is the case then gt is probably already a one hot encoding
#         if all([i == j for i, j in zip(pred.shape, gt.shape)]):
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(pred_shape)
#             if pred.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(pred.device.index)
#             y_onehot.scatter_(1, gt, 1)

#     tp = pred * y_onehot
#     fp = pred * (1 - y_onehot)
#     fn = (1 - pred) * y_onehot
    
#     dpt = compute_dpt(y_onehot)
#     dtm = 1 - dpt
#     dtm = (1+dtm).to(device)
#     weight = (dtm**2) / (1 + dtm**2)
    
#     w_fp = weight * fp
#     w_fn = (1-weight) * fn

#     tp = torch.sum(tp, axes)
#     w_fp = torch.sum(w_fp, axes)
#     w_fn = torch.sum(w_fn, axes)

#     return tp, w_fp, w_fn    
    
# class DTM_AsymLoss(nn.Module):

#     __name__= "DTM_Asym_Loss"

#     """
#     refer from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
    
#     Implementation of Asymmetric similarity loss for image segmentation task.
#     This is a special case of Tversky loss when alpha + beta = 1
#     paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         #log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         device = 'cpu'
#         #alpha: float = 0.3,
#         #gamma: float = 0.75
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
#         :param alpha: controls the penalty for false positives.
#         :param beta: penalty for false negative. Larger beta weigh recall higher
#         :param gamma:
#         """
#         super(DTM_AsymLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.device = device
#         # self.alpha = alpha
#         # self.gamma = gamma
        

#     def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = pred * mask
#                 gt = gt * mask  
#             else:
#                 pred = pred * mask.unsqueeze(1)
#                 gt = gt * mask

#         tp, w_fp, w_fn = dtm_get_tp_fp_fn_for_asym(pred, gt, axes, self.device)
#         # weight = (self.beta**2) / (1 + self.beta**2)
#         asym = (tp + self.smooth) / (tp + w_fp + w_fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 asym = asym[1:]
#             else:
#                 asym = asym[:, 1:]

#         loss = 1-asym

#         return loss.mean() 
    
# class SSLoss(nn.Module):

#     __name__= "SS_Loss"

#     """
#     refer from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
    
#     Implementation of Sensitivity-Specifity loss for image segmentation task.
#     paper: http://www.rogertam.ca/Brosch_MICCAI_2015.pdf
#     tf code: https://github.com/NifTK/NiftyNet/blob/df0f86733357fdc92bbc191c8fec0dcf49aa5499/niftynet/layer/loss_segmentation.py

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         #log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         #alpha: float = 0.3,
#         r: float = 0.1,
#         #gamma: float = 0.75
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation r
#         :param r: weight parameter in SS paper
#         """
#         super(SSLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         # self.alpha = alpha
#         self.r = r
#         # self.gamma = gamma
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = pred * mask
#                 gt = gt * mask  
#             else:
#                 pred = pred * mask.unsqueeze(1)
#                 gt = gt * mask
        
#         with torch.no_grad():
#             # if this is the case then gt is probably already a one hot encoding
#             if all([i == j for i, j in zip(pred.shape, gt.shape)]):
#                 y_onehot = gt
#             else:
#                 gt = gt.long()
#                 y_onehot = torch.zeros(pred_shape)
#                 if pred.device.type == "cuda":
#                     y_onehot = y_onehot.cuda(pred.device.index)
#                 y_onehot.scatter_(1, gt, 1)


#         # no object value
#         bg_onehot = 1 - y_onehot
#         squared_error = (y_onehot - pred)**2
#         specificity_part = torch.sum(squared_error*y_onehot, axes)/(torch.sum(y_onehot, axes)+self.smooth)
#         sensitivity_part = torch.sum(squared_error*bg_onehot, axes)/(torch.sum(bg_onehot, axes)+self.smooth)

#         loss = self.r * specificity_part + (1-self.r) * sensitivity_part

#         if not self.do_bg:
#             if self.batch_dice:
#                 loss = loss[1:]
#             else:
#                 loss = loss[:, 1:]

#         return loss.mean()
    
    
    
# def dtp_fmeasure_loss(pred, gt, axes=None, device='cpu'):
#     """
#     pred must be (B, C, H, W))
#     gt must be a label map (shape (B, 1, H, W) OR shape (B, H, W)) or one hot encoding (B, C, H, W)
#     if mask is provided it must have shape (B, 1, H, W))
#     :param pred:
#     :param gt:
#     :param axes:
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(pred.size())))

#     pred_shape = pred.shape
#     gt_shape = gt.shape

#     with torch.no_grad():
#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))

#         # if this is the case then gt is probably already a one hot encoding
#         if all([i == j for i, j in zip(pred.shape, gt.shape)]):
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(pred_shape)
#             if pred.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(pred.device.index)
#             y_onehot.scatter_(1, gt, 1)

#     tp = pred * y_onehot
#     fp = pred * (1 - y_onehot)
#     fn = (1 - pred) * y_onehot
    
#     dpt = compute_dpt(y_onehot)
#     beta_2 = (dpt*dpt).to(device)
    
#     numerator = (1 + beta_2)*tp
#     denominator = beta_2 * (tp + fn) + (tp + fp)

#     numerator = torch.sum(numerator, axes)
#     denominator = torch.sum(denominator, axes)

#     return numerator, denominator
       
# class DPT_Fmeasure_Loss(nn.Module):

#     __name__= "DPT_Fmeasure_Loss"

#     """
#     Implementation of Tversky loss for image segmentation task.
#     paper: https://arxiv.org/pdf/1706.05721.pdf

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         # log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         device = 'cpu'
#     ):
#         """
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
#         :param alpha: controls the penalty for false positives.
#         : param beta: penalty for false negative. Larger beta weigh recall higher
#         """
#         super(DPT_Fmeasure_Loss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         # self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.device = device


#     def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask

#         numerator, denominator = dtp_fmeasure_loss(pred, gt, axes, self.device)
#         w_fmeasure = (numerator + self.smooth) / (denominator + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 w_fmeasure = w_fmeasure[1:]
#             else:
#                 w_fmeasure = w_fmeasure[:, 1:]

#         loss = 1.0 - w_fmeasure


#         return loss.mean()  
    
# class log_cosh_DiceLoss(nn.Module):

#     __name__= "log_cosh_DiceLoss"

#     """
#     refer from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
    
#     Implementation of Dice loss for image segmentation task.

#     """

#     def __init__(
#         self,
#         batch_dice = False,
#         from_logits=True,
#         do_bg = True,
#         square = False,
#         log_loss=False,
#         smooth: float = 1e-7,
#         ignore_index=None,
#         eps=1e-7,
#     ):
#         """
#         :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
#         :param from_logits: If True assumes input is raw logits
#         :param smooth:
#         :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
#         :param eps: Small epsilon for numerical stability
#         :param batch_dice: sum tp, fp, fn in every batch
#         :param do_bg: inclue background in loss computation 
        
#         paper: https://arxiv.org/pdf/1606.04797.pdf

#         """
#         super(log_cosh_DiceLoss, self).__init__()

#         self.batch_dice = batch_dice
#         self.from_logits = from_logits
#         self.do_bg = do_bg
#         self.square = square
#         self.log_loss = log_loss
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.eps = eps
        

#     def forward(self, pred: Tensor, gt: Tensor, loss_mask = None) -> Tensor:
#         '''
#         pred:   shape for binary segmentation                   - B x 1 x H x W
#                 shape for multiclass/multilable segmentation    - B x C x H x W
#         gt:     shape for binary segmentation                   - B x 1 x H x W or B x H x W
#                 shape for multiclass segmentation               - B x H x W
#                 shape for multilable segmentation               - B x C x H x W (one-hot encoding)
#         '''

#         assert gt.size(0) == pred.size(0)

#         pred_shape = pred.shape
#         gt_shape = gt.shape

#         if len(pred_shape) != len(gt_shape):
#             gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))
#             gt_shape = gt.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(pred_shape)))
#         else:
#             axes = list(range(2, len(pred_shape)))

#         if self.from_logits:
#             # Apply activations to get [0..1] class probabilities
#             # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
#             # extreme values 0 and 1
#             if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#                 pred = F.logsigmoid(pred).exp()
#                 # print('sigmoid:',pred)    
#             else:
#                 pred = pred.log_softmax(dim=1).exp()
#                 # print('softmax', pred)

#         if self.ignore_index is not None:
#             mask = gt != self.ignore_index
#             pred = pred * mask
#             gt = gt * mask
#             # if all([i == j for i, j in zip(pred_shape, gt_shape)]):
#             #     pred = pred * mask
#             #     gt = gt * mask  
#             # else:
#             #     pred = pred * mask.unsqueeze(1)
#             #     gt = gt * mask

#         tp, fp, fn = get_tp_fp_fn(pred, gt, axes, loss_mask, self.square)
#         dice_score = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 dice_score = dice_score[1:]
#             else:
#                 dice_score = dice_score[:, 1:]

#         if self.log_loss:
#             dice_loss = -torch.log(dice_score.clamp_min(self.eps))
#         else:
#             dice_loss = 1.0 - dice_score
        
#         log_cosh_dice_loss = torch.log((torch.exp(dice_loss) + torch.exp(-dice_loss))/2)

#         # Dice loss is undefined for non-empty classes
#         # So we zero contribution of channel that does not have true pixels
#         # NOTE: A better workaround would be to use loss term `mean(y_pred)`
#         # for this case, however it will be a modified jaccard loss

#         return log_cosh_dice_loss.mean()


# def _lovasz_grad(gt_sorted):
#     """Compute gradient of the Lovasz extension w.r.t sorted errors
#     See Alg. 1 in paper
#     """
#     p = len(gt_sorted)
#     gts = gt_sorted.sum()
#     intersection = gts - gt_sorted.float().cumsum(0)
#     union = gts + (1 - gt_sorted).float().cumsum(0)
#     jaccard = 1.0 - intersection / union
#     if p > 1:  # cover 1-pixel case
#         jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
#     return jaccard

# #----------------------Lovasz Loss---------------------------------------------------------

# def _lovasz_hinge(logits, labels, per_image=True, ignore_index=None):
#     """
#     Binary Lovasz hinge loss
#         logits: [B, H, W] Variable, logits at each pixel (between -infinity and +infinity)
#         labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
#         per_image: compute the loss per image instead of per batch
#         ignore: void class id
#     """
#     if per_image:
#         loss = mean(
#             _lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore_index))
#             for log, lab in zip(logits, labels)
#         )
#     else:
#         loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore_index))
#     return loss


# def _lovasz_hinge_flat(logits, labels):
#     """Binary Lovasz hinge loss
#     Args:
#         logits: [P] Variable, logits at each prediction (between -iinfinity and +iinfinity)
#         labels: [P] Tensor, binary ground truth labels (0 or 1)
#         ignore: label to ignore
#     """
#     if len(labels) == 0:
#         # only void pixels, the gradients should be 0
#         return logits.sum() * 0.0
#     signs = 2.0 * labels.float() - 1.0              # scale to the range of [-1, 1]
#     errors = 1.0 - logits * Variable(signs)
#     errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
#     perm = perm.data # remove grad_fn attribution of tensor
#     gt_sorted = labels[perm]
#     grad = _lovasz_grad(gt_sorted)
#     loss = torch.dot(F.relu(errors_sorted), Variable(grad))
#     return loss


# def _flatten_binary_scores(scores, labels, ignore_index=None):
#     """Flattens predictions in the batch (binary case)
#     Remove labels equal to 'ignore'
#     """
#     scores = scores.view(-1)
#     labels = labels.view(-1)
#     if ignore_index is None:
#         return scores, labels
#     valid = labels != ignore_index
#     vscores = scores[valid]
#     vlabels = labels[valid]
#     return vscores, vlabels


# # --------------------------- MULTICLASS LOSSES ---------------------------


# def _lovasz_softmax(probas, labels, classes="present", per_image=False, ignore_index=None):
#     """Multi-class Lovasz-Softmax loss
#     Args:
#         @param probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
#         Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
#         @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
#         @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#         @param per_image: compute the loss per image instead of per batch
#         @param ignore_index: void class labels
#     """
#     if per_image:
#         loss = mean(
#             _lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore_index), classes=classes)
#             for prob, lab in zip(probas, labels)
#         )
#     else:
#         loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore_index), classes=classes)
#     return loss


# def _lovasz_softmax_flat(probas, labels, classes="present"):
#     """Multi-class Lovasz-Softmax loss
#     Args:
#         @param probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
#         @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
#         @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#     """
#     if probas.numel() == 0:
#         # only void pixels, the gradients should be 0
#         return probas * 0.0
#     C = probas.size(1)
#     losses = []
#     class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
#     for c in class_to_sum:
#         fg = (labels == c).type_as(probas)  # foreground for class c
#         if classes == "present" and fg.sum() == 0:
#             continue
#         if C == 1:
#             if len(classes) > 1:
#                 raise ValueError("Sigmoid output possible only with 1 class")
#             class_pred = probas[:, 0]
#         else:
#             class_pred = probas[:, c]
#         errors = (fg - class_pred).abs()
#         errors_sorted, perm = torch.sort(errors, 0, descending=True)
#         perm = perm.data
#         fg_sorted = fg[perm]
#         losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
#     return mean(losses)


# def _flatten_probas(probas, labels, ignore=None):
#     """Flattens predictions in the batch"""
#     if probas.dim() == 3:
#         # assumes output of a sigmoid layer
#         B, H, W = probas.size()
#         probas = probas.view(B, 1, H, W)

#     C = probas.size(1)
#     probas = torch.movedim(probas, 1, -1)  # [B, C, Di, Dj, ...] -> [B, Di, Dj, ..., C]
#     probas = probas.contiguous().view(-1, C)  # [P, C]

#     labels = labels.view(-1)
#     if ignore is None:
#         return probas, labels
#     valid = labels != ignore
#     vprobas = probas[valid]
#     vlabels = labels[valid]
#     return vprobas, vlabels


# # --------------------------- HELPER FUNCTIONS ---------------------------
# def isnan(x):
#     return x != x


# def mean(values, ignore_nan=False, empty=0):
#     """Nanmean compatible with generators."""
#     values = iter(values)
#     if ignore_nan:
#         values = ifilterfalse(isnan, values)
#     try:
#         n = 1
#         acc = next(values)
#     except StopIteration:
#         if empty == "raise":
#             raise ValueError("Empty mean")
#         return empty
#     for n, v in enumerate(values, 2):
#         acc += v
#     if n == 1:
#         return acc
#     return acc / n


# class BinaryLovaszLoss(_Loss):
#     __name__= "BinaryLovasz_Loss"
    
#     def __init__(self, per_image: bool = False, ignore_index: Optional[Union[int, float]] = None):
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.per_image = per_image

#     def forward(self, logits, target):
#         return _lovasz_hinge(logits, target, per_image=self.per_image, ignore_index=self.ignore_index)


# class LovaszLoss(_Loss):
#     __name__= "Lovasz_Loss"
#     def __init__(self, per_image=False, ignore=None):
#         super().__init__()
#         self.ignore = ignore
#         self.per_image = per_image

#     def forward(self, logits, target):
#         return _lovasz_softmax(logits, target, per_image=self.per_image, ignore_index=self.ignore)   
    
# #--------------------HausdorffDT loss---------------------------------
# class HausdorffDTLoss(nn.Module):
#     __name__= "HausdorffDT_Loss"
    
#     """Binary Hausdorff loss based on distance transform
#     Hausdorff loss implementation based on paper:
#     https://arxiv.org/pdf/1904.10030.pdf

#     copy pasted from - all credit goes to original authors:
#     https://github.com/SilmarilBearer/HausdorffLoss
#     """

#     def __init__(self, alpha=2.0, device = 'cuda:0', **kwargs):
#         super(HausdorffDTLoss, self).__init__()
#         self.alpha = alpha
#         self.device = device

#     # @torch.no_grad()
#     def distance_field(self, img: np.ndarray) -> np.ndarray:
#         field = np.zeros_like(img)

#         for batch in range(len(img)):
#             fg_mask = img[batch] > 0.5

#             if fg_mask.any():
#                 bg_mask = ~fg_mask

#                 fg_dist = edt(fg_mask)
#                 bg_dist = edt(bg_mask)

#                 field[batch] = fg_dist + bg_dist

#         return field

#     def forward(
#         self, pred: torch.Tensor, target: torch.Tensor, debug=False
#     ) -> torch.Tensor:
#         """
#         Uses one binary channel: 1 - fg, 0 - bg
#         pred: (b, 1, x, y, z) or (b, 1, x, y)
#         target: (b, 1, x, y, z) or (b, 1, x, y)
#         """
#         assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
#         assert (
#             pred.dim() == target.dim()
#         ), "Prediction and target need to be of same dimension"

#         pred = torch.sigmoid(pred)

#         pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float()
#         target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float()
#         pred_dt, target_dt = pred_dt.to(self.device), target_dt.to(self.device)

#         pred_error = (pred - target) ** 2
#         pred_error = pred_error.to(self.device)
#         distance = pred_dt ** self.alpha + target_dt ** self.alpha

#         dt_field = pred_error * distance
#         loss = dt_field.mean()

#         if debug:
#             return (
#                 loss.cpu().numpy(),
#                 (
#                     dt_field.cpu().detach().numpy()[0, 0],
#                     pred_error.cpu().detach().numpy()[0, 0],
#                     distance.cpu().detach().numpy()[0, 0],
#                     pred_dt.cpu().detach().numpy()[0, 0],
#                     target_dt.cpu().detach().numpy()[0, 0],
#                 ),
#             )

#         else:
#             return loss
    
    


# #----------------------------------------------------------Debug_losses--------------------------------------
# print('\n----------------------------------------------------------Pred_Target--------------------------------------')
# import torch
# from torch import nn
# N, C = 2, 1
# data = torch.randn(N, 16, 5, 5)
# conv = nn.Conv2d(16, C, (3, 3))
# pred = conv(data) # Shape of Batch x C x H x W
# print('pred:\n',pred.shape, '\n', pred)

# import numpy as np
# np.random.seed(0)
# arr = np.random.randint(2, size=(N, C, 3, 3))
# target = torch.from_numpy(arr).float()
# print('target:\n',target.shape, '\n',target)

# print('\n----------------------------------------------------------Debug_losses--------------------------------------')

# SoftBCEWithLogits_Loss = SoftBCEWithLogitsLoss()
# loss = SoftBCEWithLogits_Loss(pred, target)
# print('{}:'.format(SoftBCEWithLogits_Loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# WBCEWithLogitLoss_loss = WBCEWithLogitLoss(weight=1.0)
# loss = WBCEWithLogitLoss_loss(pred, target)
# print('{}:'.format(WBCEWithLogitLoss_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# WBCEWithLogitLoss_loss = WBCEWithLogitLoss(weight=0.7)
# loss = WBCEWithLogitLoss_loss(pred, target)
# print('{}:'.format(WBCEWithLogitLoss_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# WBCEWithLogitLoss_loss = WBCEWithLogitLoss(weight=1.2)
# loss = WBCEWithLogitLoss_loss(pred, target)
# print('{}:'.format(WBCEWithLogitLoss_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# BalancedBCEWithLogits_loss = BalancedBCEWithLogitsLoss()
# loss = BalancedBCEWithLogits_loss(pred, target)
# print('{}:'.format(BalancedBCEWithLogits_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# BinaryFocal_loss = BinaryFocalLoss()
# loss = BinaryFocal_loss(pred, target)
# print('{}:'.format(BinaryFocal_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# Dice_loss = DiceLoss()
# loss = Dice_loss(pred, target)
# print(DiceLoss().__name__)
# print('{}:'.format(Dice_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# Jaccard_loss = JaccardLoss()
# loss = Jaccard_loss(pred, target)
# print(JaccardLoss().__name__)
# print('{}:'.format(Jaccard_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# Tversky_loss = TverskyLoss()
# loss = Tversky_loss(pred, target)
# print(TverskyLoss().__name__)
# print('{}:'.format(Tversky_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# Asym_loss = AsymLoss()
# loss = Asym_loss(pred, target)
# print(AsymLoss().__name__)
# print('{}:'.format(Asym_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# SS_loss = SSLoss()
# loss = SS_loss(pred, target)
# print(SSLoss().__name__)
# print('{}:'.format(SS_loss), loss, '\n')

# print('----------------------------------------------------------Debug_losses--------------------------------------')
# log_cosh_Dice_loss = log_cosh_DiceLoss()
# loss = log_cosh_Dice_loss(pred, target)
# print(log_cosh_DiceLoss().__name__)
# print('{}:'.format(log_cosh_Dice_loss), loss, '\n')

