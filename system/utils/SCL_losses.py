import torch
import torch.nn as nn
import torch.nn.functional as F

# class SCLWithCosMargin(nn.Module):
#     def __init__(self, temperature=0.07, margin=0.4):
#         super(SCLWithCosMargin, self).__init__()
#         self.temperature = temperature
#         self.margin = margin

#     def forward(self, features, labels):
#         """
#         :param features: Feature representations of shape [batch_size, feature_dim]
#         :param labels: Ground truth labels of shape [batch_size]
#         :return: Supervised contrastive loss with margin
#         """
#         device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        
#         labels = labels.contiguous().view(-1, 1)
#         mask = torch.eq(labels, labels.T).float().to(device)
        
#         # Normalize the features to get cosine similarity
#         features = F.normalize(features, p=2, dim=1)
        
#         # Compute logits with margin
#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, features.T),
#             self.temperature
#         )
#         anchor_dot_contrast = anchor_dot_contrast - self.margin
        
#         # For numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
        
#         # Mask out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(features.shape[0]).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask
        
#         # Compute log-probabilities
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
#         # Compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
#         # Loss
#         loss = -mean_log_prob_pos
#         loss = loss.mean()
        
#         return loss

# class SCLWithArcMargin(nn.Module):
#     def __init__(self, temperature=0.07, margin=0.5):
#         super(SCLWithArcMargin, self).__init__()
#         self.temperature = temperature
#         self.margin = margin

#     def forward(self, features, labels):
#         """
#         :param features: Feature representations of shape [batch_size, feature_dim]
#         :param labels: Ground truth labels of shape [batch_size]
#         :return: Supervised contrastive loss with ArcMargin
#         """
#         device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        
#         labels = labels.contiguous().view(-1, 1)
#         mask = torch.eq(labels, labels.T).float().to(device)
        
#         # Normalize the features to get cosine similarity
#         features = F.normalize(features, p=2, dim=1)
        
#         # Compute cosine similarity
#         cosine_similarity = torch.matmul(features, features.T)
        
#         # Convert cosine similarity to angle and add margin
#         theta = torch.acos(cosine_similarity.clamp(-1 + 1e-7, 1 - 1e-7))
#         marginal_theta = theta + self.margin
#         cosine_with_margin = torch.cos(marginal_theta)
        
#         # Apply temperature scaling
#         logits = cosine_with_margin / self.temperature
        
#         # For numerical stability
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#         logits = logits - logits_max.detach()
        
#         # Mask out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(features.shape[0]).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask
        
#         # Compute log-probabilities
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
#         # Compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
#         # Loss
#         loss = -mean_log_prob_pos
#         loss = loss.mean()
        
#         return loss


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


class SCLWithcNCE(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.7):
        super(SCLWithcNCE, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, features, labels=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size, n_views, _ = features.shape

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            pos_mask = torch.eq(labels, labels.T).float().to(device)
        else:
            raise ValueError('Labels cannot be None!')
        
        # SupConLoss computation
        anchor_features = features[:, 0, :]  # Only the original view is used for SupConLoss
        anchor_dot_contrast = torch.div(torch.matmul(anchor_features, anchor_features.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(pos_mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        pos_mask = pos_mask * logits_mask

        # neg_mask[i, i] = 1
        neg_mask = 1 - pos_mask

        exp_logits = torch.exp(logits) * neg_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = pos_mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / mask_pos_pairs
        loss_sup = -mean_log_prob_pos.mean()

        # cNCELoss computation
        aug_features = features[:, 1, :]
        # shape: [bsz, 1]
        anchor_aug_dot_contrast = torch.div(torch.sum(anchor_features * aug_features, dim=1, keepdim=True), self.temperature)
        # print(f"anchor_aug_dot_contrast before: {anchor_aug_dot_contrast}")
        anchor_aug_dot_contrast = anchor_aug_dot_contrast - logits_max.detach()
        # print(f"shape of anchor_features: {anchor_features.shape} | shape of aug_features: {aug_features.shape} | shape of anchor_aug_dot_contrast: {anchor_aug_dot_contrast.shape}")
        # print(f"anchor_aug_dot_contrast after: {anchor_aug_dot_contrast}")
        # anchor_dot_contrast_aug = torch.einsum('bf,bvf->bv', anchor_features, features)
        # anchor_dot_contrast_aug = anchor_dot_contrast_aug - logits_max.detach()
        # anchor_dot_contrast_aug = torch.div(anchor_dot_contrast_aug[:, 1:], self.temperature)
        # aug_logits = anchor_dot_contrast_aug.sum(1) / (n_views - 1)

        pos_exp_logits = torch.exp(logits) * pos_mask
        pos_exp_logits_sum = pos_exp_logits.sum(1)
        pos_exp_logits_sum = torch.where(pos_exp_logits_sum < 1e-6, 1, pos_exp_logits_sum)
        # print(f"pos_exp_logits_sum: {pos_exp_logits_sum}")
        # loss_cNCE = valid_aug_logits - torch.log(valid_pos_exp_logits_sum)
        loss_cNCE = anchor_aug_dot_contrast - torch.log(pos_exp_logits_sum)
        # print(f"loss_sup: {loss_sup}")
        # print(f"loss_cNCE: {loss_cNCE}")

        loss_cNCE = -loss_cNCE.mean()

        # Combine losses
        # print(f"loss_sup: {loss_sup}")
        # print(f"loss_cNCE: {loss_cNCE}")
        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_cNCE

        return loss

class SCLWithMask(nn.Module):
    def __init__(self, temperature=0.07, epsilon=0.7):
        super(SCLWithMask, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, f_dim], at least 2 dimensions are required')
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
        cosine_similarity = torch.matmul(features, features.T)
        anchor_dot_contrast = torch.div(
            cosine_similarity,
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

        # Apply the similarity threshold mask
        sim_mask = (cosine_similarity <= self.epsilon).float().to(device)
        mask = mask * sim_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Adjust the normalization factor
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # Avoid division by zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


class SCLWithArcMargin(nn.Module):
    def __init__(self, temperature=0.07, margin=0.5):
        super(SCLWithArcMargin, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

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

        # compute cosine similarity
        cosine_similarity = torch.matmul(features, features.T)
        
        # Convert cosine similarity to angle and add margin
        theta = torch.acos(cosine_similarity.clamp(-1 + 1e-7, 1 - 1e-7))
        marginal_theta = theta + self.margin
        cosine_with_margin = torch.cos(marginal_theta)
        
        # Apply temperature scaling
        logits = cosine_with_margin / self.temperature
        
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


class SCLWithCosMargin(nn.Module):
    def __init__(self, temperature=0.07, margin=0.4):
        super(SCLWithCosMargin, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

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

        # compute logits with margin
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        positive_anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T) - self.margin,
            self.temperature
        )

        # for numerical stability
        positive_logits_max, _ = torch.max(positive_anchor_dot_contrast, dim=1, keepdim=True)
        positive_logits = positive_anchor_dot_contrast - positive_logits_max.detach()

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

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = positive_logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss

class SCLWithEnhancedCosMargin(nn.Module):
    def __init__(self, temperature=0.07, positive_margin=0.2, negative_margin=0.5):
        super(SCLWithEnhancedCosMargin, self).__init__()
        self.temperature = temperature
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

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

        # compute logits with margin
        cosine_similarity = torch.matmul(features, features.T)
        positive_anchor_dot_contrast = torch.div(
            cosine_similarity + self.positive_margin,
            self.temperature
        )
        negative_anchor_dot_contrast = torch.div(
            cosine_similarity - self.negative_margin,
            self.temperature
        )
        # for numerical stability
        positive_logits_max, _ = torch.max(positive_anchor_dot_contrast, dim=1, keepdim=True)
        positive_logits = positive_anchor_dot_contrast - positive_logits_max.detach()

        negative_logits_max, _ = torch.max(negative_anchor_dot_contrast, dim=1, keepdim=True)
        negative_logits = negative_anchor_dot_contrast - negative_logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(negative_logits) * logits_mask
        log_prob = positive_logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss
    
class RelaxedCL(nn.Module):
    """Supervised Contrastive Learning with enhanced loss function."""
    def __init__(self, temperature=0.07, beta=1.0, intra_class_similarity_threshold=0.7):
        super(RelaxedCL, self).__init__()
        self.temperature = temperature
        self.beta = beta
        self.intra_class_similarity_threshold = intra_class_similarity_threshold

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss.
        
        Args:
            features: [bsz, f_dim]
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, f_dim], at least 2 dimensions are required')
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

        # Mask-out self-contrast cases
        # logits_mask 是样本自己对比的位置为0，用于分母
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        # 这里的 mask 是得到同类样本（除掉自身）对应位置为1的mask，用于分子
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 对每一行进行求和，结果是一个向量，表示每个样本的类内样本数量（包括自己）
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # supervised contrastive loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        # intra-class similarity constraint using cosine similarity
        intra_class_loss = 0.0
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask_label = (labels == label).squeeze()
            if mask_label.sum() > 1:  # ensure there are at least two samples with the same label
                features_label = features[mask_label]
                cosine_similarities = torch.matmul(features_label, features_label.T)
                # only consider the upper triangular part of the similarity matrix, excluding the diagonal
                upper_triangular = torch.triu(cosine_similarities, diagonal=1)
                # add the similarity values that are above the threshold
                intra_class_loss += upper_triangular[upper_triangular > self.intra_class_similarity_threshold].sum()

        # combine losses
        total_loss = loss + intra_class_loss / batch_size
        
        return total_loss


# class SCLWithPrototypePenalty(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, penalty_weight=1.0):
#         super(SCLWithPrototypePenalty, self).__init__()
#         self.temperature = temperature
#         self.penalty_weight = penalty_weight
#     def forward(self, features, labels=None, mask=None):
#         """Compute loss for model. If both `labels` and `mask` are None,
#         it degenerates to SimCLR unsupervised loss:
#         https://arxiv.org/pdf/2002.05709.pdf

#         Args:
#             # features: [bsz, f_dim]
#             labels: ground truth of shape [bsz].
#             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         if len(features.shape) < 2:
#             raise ValueError('`features` needs to be [bsz, f_dim],'
#                              'at least 2 dimensions are required')
#         if len(features.shape) > 2:
#             features = features.view(features.shape[0], -1)

#         batch_size = features.shape[0]
#         if labels is not None and mask is not None:
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)

#         features = F.normalize(features, dim=1)

#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, features.T),
#             self.temperature)
        
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask

#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         mask_pos_pairs = mask.sum(1)
#         mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

#         # loss
#         loss = -mean_log_prob_pos
#         loss = loss.mean()

#         # Compute penalty
#         if labels is not None:
#             unique_labels = torch.unique(labels)
#             class_prototypes = []
#             for lbl in unique_labels:
#                 class_mask = (labels == lbl).squeeze()
#                 class_features = features[class_mask]
#                 class_prototype = class_features.mean(dim=0)
#                 class_prototypes.append(class_prototype)
#             class_prototypes = torch.stack(class_prototypes)

#             penalty = 0
#             for lbl in unique_labels:
#                 class_mask = (labels == lbl).squeeze()
#                 class_features = features[class_mask]
#                 class_prototype = class_prototypes[unique_labels == lbl].squeeze()
#                 class_similarities = F.cosine_similarity(class_features, class_prototype.unsqueeze(0), dim=1)
#                 penalty += class_similarities.sum()

#             penalty = penalty / batch_size
#             loss += self.penalty_weight * penalty

#         return loss

class SCLWithPrototypePenalty(nn.Module):
    """Supervised Contrastive Loss with Penalty for intra-class feature similarity."""
    def __init__(self, temperature=0.07, penalty_weight=1):
        super(SCLWithPrototypePenalty, self).__init__()
        self.temperature = temperature
        self.penalty_weight = penalty_weight

    def forward(self, features, labels=None, mask=None, class_prototypes=None):
        """Compute loss for model with penalty for intra-class feature similarity.

        Args:
            features: [bsz, f_dim] - feature vectors.
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                  has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, f_dim], at least 2 dimensions are required')
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

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # SCL Loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        # Compute penalty
        unique_labels = torch.unique(labels)
        if class_prototypes is not None:
            penalty = 0
            for i in range(len(unique_labels)):
                lbl = unique_labels[i]
                class_mask = (labels == lbl).squeeze()
                class_features = features[class_mask]
                class_prototype = class_prototypes[i]
                class_similarities = F.cosine_similarity(class_features, class_prototype.unsqueeze(0), dim=1)
                penalty += class_similarities.sum()

            penalty = penalty / batch_size
            loss += self.penalty_weight * penalty

        return loss
# 进行knn搜索,得到每个样本的邻居索引(用余弦相似度）
def knn_search(features, neighbors_num):

    dot_similarity = torch.matmul(features, features.T)

    mask = torch.ones_like(dot_similarity)
    mask.fill_diagonal_(float('-inf'))

    dot_similarity = dot_similarity * mask

    _, knn_index = dot_similarity.topk(neighbors_num, largest=True)

    return knn_index

def NCACrossEntropy(knn_indexs, pro_feat, y, temperature=1):
    batch_size = pro_feat.shape[0]
    # shape of knn_sets: [bsz, k_num, f_dim]
    knn_sets = torch.index_select(pro_feat, 0, torch.tensor(knn_indexs.flatten()).to(pro_feat.device)).view(
        len(pro_feat), -1, pro_feat.shape[1])

    knn_labels = y[knn_indexs]
 
    # 计算余弦相似度
    cos_similarities = F.cosine_similarity(knn_sets, pro_feat.unsqueeze(1), dim=2)   # 将 pro_feat 的形状从 [batch_size, feature_dim] 调整为 [batch_size, 1, feature_dim]
    # print("cos_similarities shape:", cos_similarities.shape)
    sim_exp = torch.exp(cos_similarities / temperature)
    # 指示邻居中与样本自身同类的邻居
    same = torch.eq(y.view(-1, 1), knn_labels)
    # print("same shape:", same.shape)
    # print("same:", same)
    # 当前批次中每个样本被正确分类的概率之和
    correct_pro = torch.mul(sim_exp, same.float()).sum(dim=1)
    # 当前批次中每个样本所有可能分类的指数和
    total_pro = sim_exp.sum(dim=1)

    prob = torch.div(correct_pro, total_pro)
    prob_masked = torch.masked_select(prob, prob.ne(0))  # 除去值为0的

    loss = prob_masked.log().sum(0)

    return - loss / batch_size