import torch
import torch.nn as nn

class CIAM(nn.Module):
    '''
        
    def __init__(self, in_channels, num_heads=8, embedding_dim=256):
      super(CIAM, self).__init__()
      self.embedding_dim = embedding_dim
      if isinstance(in_channels, int):  # If it's an int, make it a list
          in_channels = [in_channels]
      # Feature Aggregation
      self.feature_aggregation = nn.ModuleList([
        nn.Conv2d(c, embedding_dim, kernel_size=1) for c in in_channels
      ])
      # Transformer Encoder
      self.transformer_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
        num_layers=6
      )

    def forward(self, features):
      # Aggregate features from different scales
      aggregated_features = [
        f(features[i]) for i, f in enumerate(self.feature_aggregation)
      ]
      # Reshape and concatenate features
      batch_size = features[0].shape[0]
      aggregated_features = [
        f.flatten(2).permute(2, 0, 1) for f in aggregated_features
      ]
      aggregated_features = torch.cat(aggregated_features, dim=0)
      # Apply transformer encoder
      encoded_features = self.transformer_encoder(aggregated_features)
      # Reshape and split features
      encoded_features = encoded_features.permute(1, 2, 0).reshape(
        batch_size, self.embedding_dim, *features[0].shape[-2:]
      )
      # Return augmented features
      return encoded_features
    '''
    def __init__(self, channels, reduction_ratio=8, num_heads=32):
        super(CIAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_dim = channels // reduction_ratio
        self.fc1 = nn.Linear(channels, hidden_dim)  # Corrected input dimension
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, channels)  # Corrected output dimension
        self.sigmoid = nn.Sigmoid()
        
        # Multi-Head Self-Attention
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.query = nn.Linear(self.head_dim, self.head_dim) # Correcting this as well
        self.key = nn.Linear(self.head_dim, self.head_dim) # Correcting this as well
        self.value = nn.Linear(self.head_dim, self.head_dim) # Correcting this as well
        
        self.out = nn.Linear(channels, channels)
        

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).view(b, c))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).view(b, c))))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        # Multi-Head Self-Attention
        q = self.query(x).view(b, self.num_heads, self.head_dim, -1)
        k = self.key(x).view(b, self.num_heads, self.head_dim, -1)
        v = self.value(x).view(b, self.num_heads, self.head_dim, -1)

        attention = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(b, c, x.shape[2], x.shape[3])
        
        out = self.out(out) + x  # Residual connection

        out = out * channel_att
        
        return out
