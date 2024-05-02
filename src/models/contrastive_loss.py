import numpy as np

import torch
import torch.nn as nn


class SoftNearestNeighborsLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()

        self.temperature = temperature
    
    def forward(self, candidates, labels):
        """
        Calculate the distance between each pair of candidates. 
        Pairs with the same label are considered positive,while pairs with different labels are negative.

        Arguements:
            candidates (torch.Tensor): A tensor representing the candidates to evaluate for contrastive loss.
                                       Each candidate is expected to have associated positives and negatives
                                       from the other candidates. The tensor shape is (B, C), where B is the
                                       batch size and C represents candidate features.
            labels (torch.Tensor): A tensor of (domain) labels for each candidate, with shape (B), where B is the batch size.
        Return:
            loss (torch.Tensor)
        """
        if len(candidates) != len(labels):
            raise ValueError(f"There are {len(candidates)} candidates, but only {(len(labels))} labels")
        device = candidates.device
        b, embed_dim = candidates.shape

        scale = embed_dim**-0.5 
        
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device).float()
        mask.fill_diagonal_(0)

        distance_matrix = torch.cdist(candidates, candidates, p=2) ** 2 
        exp_distance_matrix = torch.exp(-distance_matrix * scale / self.temperature) 
        
        numerators = (exp_distance_matrix * mask).sum(dim=1)
        denominators = exp_distance_matrix.sum(dim=1) 

        # Remove the candidates that has no positive
        indices = numerators.nonzero()
        numerators = numerators[indices]
        denominators = denominators[indices]

        r = torch.log(numerators / denominators)
        loss = -r.mean()

        return loss
    

_CONTRASTIVE_LOSS = {
    "SoftNearestNeighborsLoss": SoftNearestNeighborsLoss,
}

