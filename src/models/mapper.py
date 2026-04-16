"""Mapper neural network for query embedding adaptation."""

import torch
import torch.nn as nn

from src.config import get_config, get_device


class Mapper(nn.Module):
    """Linear mapping layer to adapt query embeddings for better retrieval.
    
    This module learns a linear transformation of query embeddings to improve
    their alignment with passage embeddings in the vector space.
    """
    
    def __init__(self, embedding_dim: int = None, device: str = None):
        """Initialize the Mapper.
        
        Args:
            embedding_dim: Dimension of input/output embeddings. Uses config if None.
            device: Device to place the model on. Uses config if None.
        """
        super().__init__()
        
        config = get_config()
        self.embedding_dim = embedding_dim or config.model.embedding_dim
        self.device = device or get_device()
        
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reset()
        self.to(self.device)
    
    def reset(self):
        """Reset weights to near-identity initialization."""
        with torch.no_grad():
            self.linear.weight.data = 0.9 * torch.eye(self.embedding_dim).to(self.device)
            self.linear.bias.zero_()
    
    def forward(self, batch_query_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation to query embeddings.
        
        Args:
            batch_query_embeddings: Query embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            Transformed embeddings of shape (batch_size, embedding_dim)
        """
        return self.linear(batch_query_embeddings)
    
    def save(self, path: str):
        """Save model weights to file."""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> "Mapper":
        """Load model weights from file."""
        mapper = cls(**kwargs)
        mapper.load_state_dict(torch.load(path, map_location=mapper.device))
        return mapper
