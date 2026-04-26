import torch
import torch.nn as nn
import torchvision.models as models

class BehaviorCNNLSTM(nn.Module):
    """
    Spatio-Temporal model for cow behavior classification.
    Takes a sequence of frames or a combination of image + sensor window.
    """
    def __init__(self, hidden_dim=256, num_layers=2, num_classes=6):
        super(BehaviorCNNLSTM, self).__init__()
        
        # Spatial Feature Extractor (CNN)
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 512, 1, 1)
        
        # Temporal Modeling (LSTM)
        # Assuming sensor features or flattened visual features are sequential
        self.lstm = nn.LSTM(
            input_size=512, # From ResNet18
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(hidden_dim/2), num_classes)
        )
        
    def forward(self, x_seq):
        """
        x_seq shape: (Batch, Seq_Len, C, H, W)
        """
        b, seq_len, c, h, w = x_seq.size()
        
        # Reshape to push through CNN
        x_cnn_in = x_seq.view(b * seq_len, c, h, w)
        cnn_features = self.cnn(x_cnn_in)
        
        # Reshape back to sequence
        cnn_features = cnn_features.view(b, seq_len, -1)
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(cnn_features)
        
        # Classify based on the last hidden state
        out = self.classifier(hn[-1])
        return out
