import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            # features: [bsz, f_dim]
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, f_dim],'
                             'at least 2 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        features = F.normalize(features, dim=1)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        neg_mask = 1 - mask

        # compute log_prob
        exp_logits = torch.exp(logits) * neg_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss

class SCLWithUniversum(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07):
        super(SCLWithUniversum, self).__init__()
        self.temperature = temperature

    def forward(self, features, universum_features, labels=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            # features: [bsz, f_dim]
            universum_features: [bsz, f_dim]
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, f_dim],'
                             'at least 2 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)

        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            pos_mask = torch.eq(labels, labels.T).float().to(device)
        else:
            raise ValueError('Labels cannot be None!')

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        
        anchor_universum_dot_contrast = torch.div(
            torch.matmul(features, universum_features.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_universum_dot_contrast, dim=1, keepdim=True)
        uni_logits = anchor_universum_dot_contrast - logits_max.detach()
        ori_logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(pos_mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        pos_mask = pos_mask * logits_mask

        # neg_mask[i, i] = 1
        neg_mask = 1 - pos_mask
        exp_logits = torch.exp(ori_logits) * neg_mask

        # compute log_prob
        log_prob = uni_logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = pos_mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        # loss = -log_prob.sum(1).mean()
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss
