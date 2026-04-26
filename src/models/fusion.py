import torch
import torch.nn as nn

class MultiModalFusion(nn.Module):
    """
    Fuses visual embeddings (from ViT) and sensor window averages
    via Cross-Attention to output a unified health score representation.
    """
    def __init__(self, visual_dim=512, sensor_dim=30, hidden_dim=256, num_heads=4):
        super(MultiModalFusion, self).__init__()
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
        
        # Standard MultiheadAttention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        # Unified Health Score Predictor
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), 1),
            nn.Sigmoid() # Health score localized between 0 and 1
        )
        
    def forward(self, vis_embed, sensor_feats):
        """
        vis_embed shape: (B, visual_dim)
        sensor_feats shape: (B, sensor_dim) -> e.g., averaged window or flattened state
        """
        # (B, 1, hidden_dim) forms queries, keys, values
        query = self.visual_proj(vis_embed).unsqueeze(1)
        key_value = self.sensor_proj(sensor_feats).unsqueeze(1)
        
        # Attend to sensor data using visual features as query
        attn_output, attn_weights = self.cross_attention(query, key_value, key_value)
        
        # Squeeze temporal/sequence dim -> (B, hidden_dim)
        fused_representation = attn_output.squeeze(1)
        
        # Output prediction
        health_score = self.score_head(fused_representation)
        
        return health_score, fused_representation, attn_weights
