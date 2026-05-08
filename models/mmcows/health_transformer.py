import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class HealthRiskTransformer(nn.Module):
    """
    Temporal Transformer for health risk score regression from Task 5.
    """
    def __init__(self, in_dim: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, dropout: float = 0.2,
                 forecast_h: int = 24):
        super().__init__()
        self.d_model    = d_model
        self.forecast_h = forecast_h

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))

        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers,
            enable_nested_tensor=False)

        self.feat_attn = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.Sigmoid())

        def _head(out_dim: int):
            return nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model // 2, out_dim),
                nn.Sigmoid())

        self.score_now_head    = _head(1)
        self.score_future_head = _head(forecast_h)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        feat_w  = self.feat_attn(x.mean(dim=1))
        x_gated = x * feat_w.unsqueeze(1)
        h   = self.pos_enc(self.input_proj(x_gated))
        seq = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        enc = self.encoder(seq)
        cls = enc[:, 0]
        score_now    = self.score_now_head(cls)    * 100
        score_future = self.score_future_head(cls) * 100
        return score_now, score_future, feat_w
