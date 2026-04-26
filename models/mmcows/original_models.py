import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models

# --- MODULE 1: COW IDENTIFICATION & EMBEDDINGS ---
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * torch.cos(torch.tensor(self.m)) - sine * torch.sin(torch.tensor(self.m))
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return output

class CowReIDModel(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=16, pretrained=False, embedding_dim=512):
        super(CowReIDModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
        self.fc = nn.Linear(self.backbone.num_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.arcface = ArcFace(embedding_dim, num_classes)
        
    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.bn(self.fc(features))
        if labels is not None:
            return self.arcface(embeddings, labels), embeddings
        return embeddings

# --- MODULE 2: BEHAVIOR ANALYSIS (USED FOR HEAT STRESS) ---
class BehaviorCNNLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, num_classes=6):
        super(BehaviorCNNLSTM, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]) # Output: 512
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x_seq):
        """
        x_seq: (Batch, Seq_Len, C, H, W) or (Batch, Seq_Len, Features)
        If Features (e.g. 512), skip CNN.
        """
        if x_seq.ndim == 5:
            b, seq_len, c, h, w = x_seq.size()
            x_cnn_in = x_seq.view(b * seq_len, c, h, w)
            cnn_features = self.cnn(x_cnn_in).view(b, seq_len, -1)
        else:
            cnn_features = x_seq
            
        lstm_out, (hn, cn) = self.lstm(cnn_features)
        return self.classifier(hn[-1])

# --- MODULE 3: MILK PRODUCTIVITY PREDICTION ---
class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim=30, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.regressor = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        
    def forward(self, src):
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.regressor(output.mean(dim=1))

# --- MODULE 4: ANOMALY DETECTION ---
class SensorAutoencoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=64):
        super(SensorAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(True),
            nn.Linear(256, 128), 
            nn.ReLU(True), 
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), 
            nn.ReLU(True),
            nn.Linear(128, 256), 
            nn.ReLU(True), 
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def compute_anomaly_score(self, x):
        reconstructed = self.forward(x)
        mse_loss = nn.MSELoss(reduction='none')(reconstructed, x)
        return mse_loss.mean(dim=1)

# --- MODULE 5: MULTI-MODAL FUSION ---
class MultiModalFusion(nn.Module):
    def __init__(self, visual_dim=512, sensor_dim=30, hidden_dim=256, num_heads=4):
        super(MultiModalFusion, self).__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, vis_embed, sensor_feats):
        query = self.visual_proj(vis_embed).unsqueeze(1)
        key_value = self.sensor_proj(sensor_feats).unsqueeze(1)
        attn_output, weights = self.cross_attention(query, key_value, key_value)
        return self.score_head(attn_output.squeeze(1)), attn_output.squeeze(1), weights

# --- MODULE 6: HEAT STRESS PREDICTION ---
class HeatStressTransformer(nn.Module):
    def __init__(self, in_dim=19, d_model=128, n_heads=8, n_layers=4,
                 dropout=0.2, num_classes=4, forecast_h=6):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        
        # Simple PE for HeatStress
        self.pos_enc = nn.Parameter(torch.zeros(1, 100, d_model))
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model)*0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        
        def _head(out):
            return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model,d_model//2),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2,out))
        self.cls_head      = _head(num_classes)
        self.forecast_head = _head(forecast_h)
        self.risk_head     = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model,64),
                                            nn.GELU(), nn.Linear(64,1), nn.Sigmoid())

    def forward(self, x):
        B, T, _ = x.size()
        h = self.input_proj(x) + self.pos_enc[:, :T]
        seq = torch.cat([self.cls_token.expand(B,-1,-1), h], dim=1)
        out = self.encoder(seq)
        c = out[:,0]
        return self.cls_head(c), self.forecast_head(c), self.risk_head(c)
