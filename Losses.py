import torch
import torch.nn as nn
import torch.nn.functional as F


class OCC_loss(nn.Module):
    def __init__(self, margin=1.0):
        super(OCC_loss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels, center):

        dist = torch.norm(embeddings - center, p=2, dim=1)  # [B]
        pos_loss = labels * dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        loss = (pos_loss + neg_loss).mean()
        return loss





class MultiCenterOCC_loss(nn.Module):
    """
    Soft Multi-Center One-Class Classification loss (mixture-of-hyperspheres
    with learnable, temperature-annealed responsibility).

    Instead of a single global center, we maintain K learnable centers. Rather
    than hard-assigning each embedding to its single nearest center (which only
    back-propagates through the "winning" center and starves the others of
    gradient, i.e. plain k-means/VQ behavior), every embedding is given a SOFT
    responsibility distribution over all K centers via a softmax over negative
    distances:

        weights_k = softmax_k( -dist(x, center_k) / temperature )
        soft_dist = sum_k weights_k * dist(x, center_k)

    This means every center receives a (weighted) gradient from every sample
    in the batch, eliminating "dead" centers and smoothing the loss surface
    near cluster boundaries.

    The temperature has two components:
        1. A learnable scalar `log_temperature` (so the model can decide how
           soft/hard the routing should be, jointly with the rest of training).
        2. An external `anneal_factor` (0 < anneal_factor <= 1) supplied by the
           caller (typically from the training loop / model's global step) that
           multiplies the learnable temperature, letting training start soft
           (stable, exploratory) and gradually sharpen towards hard routing
           (specialized centers) as training progresses.

    in-class samples (label==1) are pulled toward centers (weighted by
    responsibility); out-of-class samples (label==0) are pushed away from
    centers (weighted by responsibility) until at least `margin` away.

    An optional center-separation regularizer (`center_sep_weight`) keeps the
    K centers from collapsing onto one another.
    """
    def __init__(self, margin=1.0, center_sep_weight=0.0, init_temperature=1.0,
                 min_temperature=0.05):
        super(MultiCenterOCC_loss, self).__init__()
        self.margin = margin
        self.center_sep_weight = center_sep_weight
        self.min_temperature = min_temperature
        # learnable temperature, parameterized in log-space to keep it positive
        self.log_temperature = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_temperature)))))

    def get_temperature(self, anneal_factor=1.0):
        temp = torch.exp(self.log_temperature) * anneal_factor
        return torch.clamp(temp, min=self.min_temperature)

    def soft_assign(self, embeddings, centers, anneal_factor=1.0):
        """
        embeddings: [B, D], centers: [K, D]
        Returns:
            soft_dist  [B]    - responsibility-weighted distance to centers
            weights    [B, K] - soft responsibility distribution per sample
            dist_matrix[B, K] - raw euclidean distances
        """
        dist_matrix = torch.cdist(embeddings, centers, p=2)          # [B, K]
        temperature = self.get_temperature(anneal_factor)
        weights = F.softmax(-dist_matrix / temperature, dim=1)        # [B, K]
        soft_dist = (weights * dist_matrix).sum(dim=1)                # [B]
        return soft_dist, weights, dist_matrix

    def forward(self, embeddings, labels, centers, anneal_factor=1.0):
        soft_dist, weights, dist_matrix = self.soft_assign(embeddings, centers, anneal_factor)

        pos_loss = labels * soft_dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - soft_dist).pow(2)
        loss = (pos_loss + neg_loss).mean()

        if self.center_sep_weight > 0 and centers.size(0) > 1:
            center_dist = torch.cdist(centers, centers, p=2)
            K = centers.size(0)
            off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=centers.device)
            # encourage centers to be at least `margin` apart from each other
            sep_loss = F.relu(self.margin - center_dist[off_diag_mask]).pow(2).mean()
            loss = loss + self.center_sep_weight * sep_loss

        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        """
        Computes the InfoNCE loss.
        Args:
            features1: output of the first model (batch_size, feature_dim)
            features2: output of the second model (batch_size, feature_dim)
        Returns:
            loss: the InfoNCE loss value
        """
        batch_size = features1.size(0)

        similarity_matrix = torch.mm(features1, features2.T) / self.temperature
        mask = torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        positives = similarity_matrix[mask].view(batch_size, 1)
        negatives = similarity_matrix[~mask].view(batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=similarity_matrix.device)
        loss = F.cross_entropy(logits, labels)

        return loss
