import torch
import torch.nn as nn

class SensorAutoencoder(nn.Module):
    """
    Unsupervised Anomaly Detection utilizing an Autoencoder architecture.
    Compresses and reconstructs fused mult-modal inputs. High reconstruction loss indicates anomaly.
    """
    def __init__(self, input_dim=512, latent_dim=64):
        super(SensorAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder
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
        """
        Returns reconstruction error per sample in batch.
        """
        reconstructed = self.forward(x)
        mse_loss = nn.MSELoss(reduction='none')(reconstructed, x)
        anomaly_score = mse_loss.mean(dim=1)  # average over feature dimension
        return anomaly_score
