import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ArcFace(nn.Module):
    """ArcFace margin product."""
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # 1. Normalize input and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. Calculate theta and add margin
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * torch.cos(torch.tensor(self.m)) - sine * torch.sin(torch.tensor(self.m))
        
        # 3. Apply margin to true class
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 4. Scale
        output *= self.s
        return output

class CowReIDModel(nn.Module):
    """
    Cow Re-Identification Model using ViT/Swin + ArcFace.
    Extracts visual features for individual cow recognition.
    """
    def __init__(self, model_name='vit_base_patch16_224', num_classes=16, pretrained=True, embedding_dim=512):
        super(CowReIDModel, self).__init__()
        # Backbone (ViT or Swin)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0) # num_classes=0 extracts features
        
        # Head
        self.fc = nn.Linear(self.backbone.num_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # Metric Learning Loss Function
        self.arcface = ArcFace(embedding_dim, num_classes)
        
    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.bn(self.fc(features))
        
        if labels is not None:
            # Training mode: return logits with ArcFace margin applied
            logits = self.arcface(embeddings, labels)
            return logits, embeddings
        else:
            # Inference mode: return just embeddings for FAISS search
            return embeddings
