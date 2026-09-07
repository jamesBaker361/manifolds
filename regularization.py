

import torch
import torch.nn.functional as F

def graph_laplacian(activations, neighbor_activations):
    diffs = activations.unsqueeze(1) - neighbor_activations
    return (diffs ** 2).sum()

def contractive(activations, dictionary): #based off of https://icml.cc/2011/papers/455_icmlpaper.pdf
    dh_sq = (activations * (1-activations)) ** 2
    atom_norms_sq = (dictionary ** 2).sum(dim=1)
    return (dh_sq * atom_norms_sq).sum()

def cosine_constrastive(activations, other_activations, data,other_data): #weight the difference by differences in cosine distance between data
    cosine_weight = F.cosine_similarity(data.unsqueeze(1),other_data,dim=-1)
    diffs = activations.unsqueeze(1) - other_activations
    diffs = cosine_weight.unsqueeze(-1) * diffs
    return (diffs**2).sum()