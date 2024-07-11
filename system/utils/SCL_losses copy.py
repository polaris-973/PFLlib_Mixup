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
            negative_mask = 1 - mask
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

        # compute log_prob
        exp_logits = torch.exp(logits) * negative_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss

class SCLWithcNCE(nn.Module):
    def __init__(self, temperature=0.07, alpha=0):
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

        batch_size = features.shape[0]
        n_views = features.shape[1]
        labels = labels.contiguous().view(-1, 1)

        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        mask = torch.eq(labels, labels.T).float().to(device)

        # SupConLoss computation
        features_anchor = features[:, 0, :]  # Only the first view is used for SupConLoss
        anchor_dot_contrast = torch.div(torch.matmul(features_anchor, features_anchor.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # print(f"logits: {logits}")
        # exp_logits = torch.exp(logits) * logits_mask
        negative_mask = 1 - mask
        exp_logits = torch.exp(logits) * negative_mask
        # print(f"exp_logits: {exp_logits}")
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss_sup = -mean_log_prob_pos.mean()

        ls, counts = torch.unique(labels, return_counts=True)
        # print(f"flag - labels and label_counts: {ls}, {counts}")
        # if torch.isnan(loss_sup):
        #     print(f"labels: {labels.T}")
        #     print(f"mask: {mask}")
        #     print(f"negative_mask: {negative_mask}")
        #     print(f"features: {features}")
        #     print(f"features_anchor: {features_anchor}")
        #     print(f"anchor_dot_contrast: {anchor_dot_contrast}")
        #     # print(f"logits_max: {logits_max}")
        #     print(f"logits: {logits}")
        #     print(f"torch.exp(logits): {torch.exp(logits)}")
        #     print(f"exp_logits: {exp_logits}")
        #     print(f"exp_logits.sum(1, keepdim=True): {exp_logits.sum(1, keepdim=True).T}")
        #     print(f"torch.log(exp_logits.sum(1, keepdim=True)): {torch.log(exp_logits.sum(1, keepdim=True)).T}")
        #     print(f"log_prob: {log_prob}")
        #     print(f"mask_pos_pairs: {mask_pos_pairs}")
        #     print(f"mean_log_prob_pos: {mean_log_prob_pos}")
        # #     return
        # print(f"loss_sup: {loss_sup}")

        # L_cNCE computation using original and augmented features
        # print(f"features_anchor: {features_anchor}")
        anchor_dot_contrast_aug = torch.einsum('bf,bvf->bv', features_anchor, features)
        anchor_dot_contrast_aug = anchor_dot_contrast_aug - logits_max.detach()
        # if torch.isnan(anchor_dot_contrast_aug).any():
        # print(f"shape of anchor_dot_contrast_aug: {anchor_dot_contrast_aug.shape}")
        # print(f"anchor_dot_contrast_aug before 0: {anchor_dot_contrast_aug[:, 0:].shape}")
        # print(f"anchor_dot_contrast_aug before 1: {anchor_dot_contrast_aug[:, 1:].shape}")
        anchor_dot_contrast_aug = torch.div(anchor_dot_contrast_aug[:, 1:], self.temperature)
        # print(f"before - shape of anchor_dot_contrast_aug: {anchor_dot_contrast_aug.shape}")
        # print(f"before - anchor_dot_contrast_aug: {anchor_dot_contrast_aug}")
        # anchor_dot_contrast_aug = torch.div(anchor_dot_contrast_aug[:, 1:], self.temperature)
        # print(f"after - shape of anchor_dot_contrast_aug: {anchor_dot_contrast_aug.shape}")
        # print(f"after - anchor_dot_contrast_aug: {anchor_dot_contrast_aug}")
        # print(f"anchor_dot_contrast_aug after: {anchor_dot_contrast_aug}")
        aug_logits = anchor_dot_contrast_aug.sum(1) / (n_views - 1)
        # print(f"aug_logits: {aug_logits}")

        pos_exp_logits = torch.exp(logits) * mask

        pos_exp_logits_sum = pos_exp_logits.sum(1)
        # print(f"pos_exp_logits: {pos_exp_logits}")
        # print(f"pos_exp_logits_sum before: {pos_exp_logits_sum}")
        pos_exp_logits_sum = torch.where(pos_exp_logits_sum < 1e-6, 1, pos_exp_logits_sum)
        
        # print(f"pos_exp_logits_sum after: {pos_exp_logits_sum}")
        # print(f"pos_exp_logits_sum log after: {torch.log(pos_exp_logits_sum)}")


        # loss_cNCE = valid_aug_logits - torch.log(valid_pos_exp_logits_sum)
        loss_cNCE = aug_logits - torch.log(pos_exp_logits_sum)


        # if torch.isnan(loss_cNCE).any():
        #     print(f"avoid_div_zero_mask: {avoid_div_zero_mask}")
        #     print(f"after mask - aug_logits: {aug_logits}")
        #     print(f"pos_exp_logits_sum: {pos_exp_logits_sum}")

        # print(f"loss_cNCE: {loss_cNCE}")

        loss_cNCE = -loss_cNCE.mean()
        # print(f"loss_cNCE -mean-: {loss_cNCE}")
        # if torch.isnan(loss_cNCE):
        # #     print(f"features: {features}")
        # #     print(f"features_anchor: {features_anchor}")
        #     print(f"labels: {labels.T}")
        #     print(f"mask: {mask}")
        #     print(f"negative_mask: {negative_mask}")
        #     print(f"features: {features}")
        #     print(f"features_anchor: {features_anchor}")
        #     print(f"anchor_dot_contrast: {anchor_dot_contrast}")
        #     # print(f"logits_max: {logits_max}")
        #     print(f"logits: {logits}")
        #     print(f"torch.exp(logits): {torch.exp(logits)}")
        #     print(f"exp_logits: {exp_logits}")
        #     print(f"exp_logits.sum(1, keepdim=True): {exp_logits.sum(1, keepdim=True).T}")
        #     print(f"torch.log(exp_logits.sum(1, keepdim=True)): {torch.log(exp_logits.sum(1, keepdim=True)).T}")
        #     print(f"log_prob: {log_prob}")
        #     print(f"mask_pos_pairs: {mask_pos_pairs}")
        #     print(f"mean_log_prob_pos: {mean_log_prob_pos}")
        #     # print(f"labels: {labels}")
        #     print(f"anchor_dot_contrast_aug: {anchor_dot_contrast_aug}")
        #     print(f"aug_logits: {aug_logits}")
        #     print(f"pos_exp_logits: {pos_exp_logits}")
        #     print(f"pos_exp_logits_sum: {pos_exp_logits_sum}")

        # Combine losses
        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_cNCE
        
        # loss = loss_sup
        # print(f"loss: {loss}")

        return loss
    
# class SCLWithcNCE(nn.Module):
#     def __init__(self, temperature=0.07, alpha=0.7):
#         super(SCLWithcNCE, self).__init__()
#         self.temperature = temperature
#         self.alpha = alpha

#     def forward(self, features, labels=None):
#         """
#         Args:
#             features: hidden vector of shape [bsz, n_views, ...].
#             labels: ground truth of shape [bsz].
#         Returns:
#             A loss scalar.
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)

#         batch_size = features.shape[0]
#         n_views = features.shape[1]
#         labels = labels.contiguous().view(-1, 1)

#         if labels.shape[0] != batch_size:
#             raise ValueError('Num of labels does not match num of features')
        
#         mask = torch.eq(labels, labels.T).float().to(device)
#         negative_mask = 1 - mask

#         # SupConLoss computation
#         features_anchor = features[:, 0, :]  # Only the first view is used for SupConLoss
#         anchor_dot_contrast = torch.div(torch.matmul(features_anchor, features_anchor.T), self.temperature)

#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#         # logits = anchor_dot_contrast

#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
#         mask = mask * logits_mask

#         exp_logits = torch.exp(logits) * negative_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         mask_pos_pairs = mask.sum(1)
#         mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
#         loss_sup = -mean_log_prob_pos.mean()

#         # L_cNCE computation using original and augmented features
#         anchor_dot_contrast_aug = torch.einsum('bf,bvf->bv', features_anchor, features)
#         anchor_dot_contrast_aug = torch.div(anchor_dot_contrast_aug[:, 1:], self.temperature)

#         aug_logits = anchor_dot_contrast_aug.sum(1) / (n_views - 1)

#         pos_exp_logits = torch.exp(anchor_dot_contrast) * mask

#         pos_exp_logits_sum = pos_exp_logits.sum(1)

#         loss_cNCE = aug_logits - torch.log(pos_exp_logits_sum)

#         loss_cNCE = -loss_cNCE.mean()

#         # Combine losses
#         loss = (1 - self.alpha) * loss_sup + self.alpha * loss_cNCE

#         return loss

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
    
