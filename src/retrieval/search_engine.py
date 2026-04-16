"""FAISS-based search engine for semantic search."""

from typing import Tuple, Literal
import numpy as np
import torch
import faiss

from src.config import get_config


SimilarityMetric = Literal["L2", "IP", "CS"]


class SearchEngine:
    """FAISS-based vector search engine.
    
    Supports L2 distance, inner product (IP), and cosine similarity (CS) metrics.
    """
    
    def __init__(
        self, 
        name: str = "faiss",
        similarity_metric: SimilarityMetric = None,
        capacity: int = None,
        embedding_dim: int = None
    ):
        """Initialize the search engine.
        
        Args:
            name: Name identifier for the search engine
            similarity_metric: Distance metric to use. Uses config if None.
            capacity: Maximum number of results to return. Uses config if None.
            embedding_dim: Dimension of embeddings. Uses config if None.
        """
        config = get_config()
        
        self.name = name
        self.similarity_metric = similarity_metric or config.search_engine.similarity_metric
        self.capacity = capacity or config.search_engine.capacity
        self.embedding_dim = embedding_dim or config.model.embedding_dim
        
        # Initialize FAISS index
        if self.similarity_metric == "L2":
            self.engine = faiss.IndexFlatL2(self.embedding_dim)
        elif self.similarity_metric in ["IP", "CS"]:
            self.engine = faiss.IndexFlatIP(self.embedding_dim)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def index(self, embeddings: torch.Tensor) -> None:
        """Index embeddings for search.
        
        Args:
            embeddings: Embeddings tensor of shape (num_items, embedding_dim)
        """
        embeddings_np = self._to_numpy(embeddings)
        
        if self.similarity_metric == "CS":
            embeddings_np = self._normalize(embeddings_np)
        
        self.engine.add(embeddings_np)
    
    def search(
        self, 
        embedding: torch.Tensor, 
        k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.
        
        Args:
            embedding: Query embedding of shape (embedding_dim,) or (1, embedding_dim)
            k: Number of results to return. Uses capacity if None.
            
        Returns:
            Tuple of (distances, indices) arrays of shape (1, k)
        """
        k = k or self.capacity
        embedding_np = self._to_numpy(embedding)
        
        if embedding_np.ndim == 1:
            embedding_np = embedding_np.reshape(1, -1)
        
        if self.similarity_metric == "CS":
            embedding_np = self._normalize(embedding_np)
        
        distances, indices = self.engine.search(embedding_np, k)
        return distances, indices
    
    def reset(self):
        """Clear all indexed embeddings."""
        self.engine.reset()
    
    @property
    def num_indexed(self) -> int:
        """Return the number of indexed embeddings."""
        return self.engine.ntotal
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine similarity."""
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        return embeddings / norm
