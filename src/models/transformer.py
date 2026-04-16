"""Sentence transformer wrapper for encoding text to embeddings."""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union

from src.config import get_config, get_device


class RowTransformer:
    """Wrapper for sentence transformer models.
    
    Provides a simple interface for encoding sentences into dense embeddings
    using pre-trained sentence transformer models.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """Initialize the transformer.
        
        Args:
            model_name: Name of the sentence transformer model. Uses config if None.
            device: Device to run the model on. Uses config if None.
        """
        config = get_config()
        self.model_name = model_name or config.model.name
        self.device = device or get_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def encode_sentence(
        self, 
        sentence: Union[str, List[str]], 
        convert_to_tensor: bool = True
    ) -> torch.Tensor:
        """Encode sentence(s) into embeddings.
        
        Args:
            sentence: Single sentence or list of sentences to encode
            convert_to_tensor: Whether to return as PyTorch tensor
            
        Returns:
            Embeddings as tensor of shape (embedding_dim,) for single sentence
            or (num_sentences, embedding_dim) for multiple sentences
        """
        return self.model.encode(
            sentence, 
            convert_to_tensor=convert_to_tensor, 
            device=self.device
        )
    
    def encode_batch(
        self, 
        sentences: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> torch.Tensor:
        """Encode a batch of sentences with progress bar.
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Embeddings tensor of shape (num_sentences, embedding_dim)
        """
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            device=self.device
        )
