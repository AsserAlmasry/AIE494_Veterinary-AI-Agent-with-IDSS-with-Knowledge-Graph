import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    """
    Milk Productivity Predictor based on historical sensor and behavior data.
    """
    def __init__(self, feature_dim=30, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        
        # Project raw features to d_model space
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1) # Predict single float continuous target: yield
        )
        
    def forward(self, src):
        """
        src shape: (Batch, Seq_Len, feature_dim)
        """
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src)
        
        # Global Average Pooling over temporal dimension
        pooled_output = output.mean(dim=1)
        
        # Predict
        predicted_yield = self.regressor(pooled_output)
        return predicted_yield

class PositionalEncoding(nn.Module):
    """Standard positional encoding to inject temporal order context to Transformer"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
