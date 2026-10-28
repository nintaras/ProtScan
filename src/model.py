import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    Attention μηχανισμός που αποδίδει βάρη σπουδαιότητας (attention weights) 
    σε κάθε αμινοξύ της πρωτεϊνικής αλληλουχίας.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, Embed_Dim)
        scores = self.attention(x)  # (Batch, Seq_Len, 1)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=1)  # (Batch, Seq_Len, 1)
        pooled = torch.sum(weights * x, dim=1)  # (Batch, Embed_Dim)
        return pooled, weights.squeeze(-1)

class ProteinClassifier(nn.Module):
    """
    Downstream ταξινομητής πρωτεϊνών με Attention Pooling πάνω από τα ESM-2 embeddings.
    """
    def __init__(self, embed_dim: int = 320, num_classes: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.pooler = AttentionPooling(embed_dim)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, Embed_Dim)
        pooled_rep, attn_weights = self.pooler(x, mask)
        logits = self.classifier(pooled_rep)
        return logits, attn_weights
