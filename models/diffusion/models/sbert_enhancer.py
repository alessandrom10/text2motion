import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEnhancer(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim_multiplier: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * hidden_dim_multiplier),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * hidden_dim_multiplier, embedding_dim)
        )
        self.simple_net = nn.Sequential(nn.Linear(embedding_dim, embedding_dim))
    def forward(self, x): return x + self.simple_net(x)

class TransformerEnhancer(nn.Module):
    def __init__(self, embedding_dim: int, nhead: int = 4, num_encoder_layers: int = 1, dropout_rate: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
    def forward(self, x): return x + self.encoder(x.unsqueeze(1)).squeeze(1)

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