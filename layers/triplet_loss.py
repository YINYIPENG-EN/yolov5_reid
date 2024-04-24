# encoding: utf-8

import torch
from IPython import embed
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)  # m=bs,n=bs
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)  # a²
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()  # b^2
    dist = xx + yy  # a^2+b^2
    dist.addmm_(1, -2, x, y.t())  # a^2+b^2 - 2ab
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability  限定一下范围防止为最小出现0无法求导
    return dist  # 返回欧式距离


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)  # get batch size

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())  # 正样本ID筛选：同一个ID为True
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())  # 负样本ID筛选，不同ID为True

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)  # 计算每个anchor到最难正样本的距离和索引


    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)  # 创建一个空的tensor,shape和labels一样为bs
               .copy_(torch.arange(0, N).long())  # 给该空的tensor分配0~N
               .unsqueeze(0).expand(N, N))  # 变成NxN矩阵，每行元素为0~N-1
        # shape [N, 1]
        p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)  # 得到最难正样本的索引
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)  # 得到最难负样本的索引
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    # Triploss(Dap-Dan+α) margin就是α，ranking_loss就是计算(Dap-Dan+α)
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            # 排序损失函数，D(x1,x2,y),x1,x2是给定的待排序的两个输入，y代表真实标签∈[-1,1].当y=1，x1排在x2之前，y=-1,x1排在x2之后。
            # max(0,-y * (x1-x2)+margin)
            # x1, x 2 x2x2排序正确且− y ∗ ( x 1 − x 2 ) > margin, 则loss为0
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)

        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False, return_inds=True):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)  # 欧式距离矩阵
        if return_inds:
            dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist_mat, labels, return_inds)
        else:
            dist_ap, dist_an = hard_example_mining(dist_mat, labels, return_inds)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an if not return_inds else loss, dist_ap, dist_an, p_inds, n_inds

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


