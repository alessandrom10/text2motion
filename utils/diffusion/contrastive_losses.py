import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLossWrapper(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)      
    def forward(self, a, p, n): return self.loss(a, p, n)

class InfoNCELoss(nn.Module):
    def __init__(self, temp: float = 0.07):
        super().__init__()
        self.temp = temp
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, feats):
        if feats.shape[0] % 2 != 0:
            raise ValueError("Batch size must be even.")
        n = feats.shape[0] // 2
        feats = F.normalize(feats, p=2, dim=1)
        sim = torch.matmul(feats, feats.T) / self.temp
        sim.masked_fill_(torch.eye(2 * n, device=feats.device, dtype=torch.bool), float('-inf'))
        targets = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(feats.device)
        return self.loss_fn(sim, targets)