"""Matched encoders and action dynamics with explicitly non-detached SIGReg."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width,width), nn.BatchNorm1d(width), nn.ReLU(),
                                 nn.Linear(width,width), nn.BatchNorm1d(width))
    def forward(self,x):
        return F.relu(x+self.net(x))

class Encoder(nn.Module):
    def __init__(self, stickers, width=256, latent=64, blocks=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(stickers*6,width), nn.BatchNorm1d(width),nn.ReLU(),
                                 nn.Linear(width,latent), nn.BatchNorm1d(latent),nn.ReLU(),
                                 *[ResidualBlock(latent) for _ in range(blocks)],
                                 nn.Linear(latent,latent))
    def forward(self,state):
        x=F.one_hot(state.long(),num_classes=6).float().flatten(1)
        return self.net(x)

class GoalScorer(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4*latent,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,z,zg):
        return self.net(torch.cat([z,zg,z-zg,z*zg],dim=-1)).squeeze(-1)

class Predictor(nn.Module):
    def __init__(self,latent=64):
        super().__init__()
        self.action=nn.Embedding(12,32)
        self.net=nn.Sequential(nn.Linear(latent+32,256),nn.ReLU(),
                               nn.Linear(256,latent))
    def forward(self,z,a):
        return z+self.net(torch.cat([z,self.action(a)],dim=-1))

class CubeModel(nn.Module):
    def __init__(self,size=3,width=256,latent=64,blocks=2):
        super().__init__()
        self.encoder=Encoder(6*size*size,width,latent,blocks)
        self.scorer=GoalScorer(latent)
        self.predictor=Predictor(latent)
        self.reference_head=nn.Linear(latent,1)
        self.q_action=nn.Embedding(12,32)
        self.q_head=nn.Sequential(nn.Linear(4*latent+32,128),nn.ReLU(),nn.Linear(128,1))
    def q(self,z,zg,a):
        x=torch.cat([z,zg,z-zg,z*zg,self.q_action(a)],dim=-1)
        return self.q_head(x).squeeze(-1)

def sigreg(z,projections=64):
    """Official LeJEPA Epps-Pulley ECF statistic; average random slices.

    N is number of independent batch examples. t ∈ [0,3], 17 knots;
    reference and integration window exp(-t²/2). Both encoder branches
    receive gradients in callers. Embeddings are NOT normalized.
    See docs/JEPA_OBJECTIVE_AUDIT.md for exact source and adaptation.
    """
    z=z.float()
    direction=F.normalize(torch.randn(z.shape[-1],projections,device=z.device),dim=0)
    t=torch.linspace(0,3,17,device=z.device)
    window=torch.exp(-t.square()/2)
    weights=torch.full_like(t,2*(3/16))
    weights[0]/=2; weights[-1]/=2
    phase=(z@direction).unsqueeze(-1)*t
    real=phase.cos().mean(0)
    imag=phase.sin().mean(0)
    error=(real-window).square()+imag.square()
    return z.shape[0]*(error*window*weights).sum(-1).mean()

def parameter_counts(model,mode):
    modules={"reference":[model.encoder,model.reference_head],
             "scratch":[model.encoder,model.scorer],
             "q":[model.encoder,model.q_action,model.q_head],
             "jepa":[model.encoder,model.scorer,model.predictor],
             "shuffled":[model.encoder,model.scorer,model.predictor]}
    return sum(p.numel() for module in modules[mode] for p in module.parameters())
