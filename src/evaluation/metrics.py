"""Evaluation metrics for retrieval performance."""

from typing import List, Dict, Optional, Tuple
import torch
from dataclasses import dataclass

from src.models import RowTransformer, Mapper
from src.retrieval import SearchEngine
from src.config import get_config, get_device


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    recall: float
    mrr: float
    k: int
    
    def __str__(self) -> str:
        return f"Recall@{self.k}: {self.recall:.2f}%, MRR@{self.k}: {self.mrr:.2f}%"


def compute_recall_at_k(
    retrieved: set, 
    relevant: set, 
) -> float:
    """Compute recall at K.
    
    Args:
        retrieved: Set of retrieved item indices
        relevant: Set of relevant item indices
        
    Returns:
        Recall score as a fraction
    """
    if not relevant:
        return 0.0
    intersection = len(relevant & retrieved)
    return intersection / len(relevant)


def compute_mrr_at_k(
    retrieved_list: List[int], 
    relevant: set,
) -> float:
    """Compute Mean Reciprocal Rank at K.
    
    Args:
        retrieved_list: Ordered list of retrieved item indices
        relevant: Set of relevant item indices
        
    Returns:
        Reciprocal rank (1/rank of first relevant item, or 0)
    """
    for rank, item in enumerate(retrieved_list, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    dataset: List[Dict],
    row_strings: List[str],
    k: int = 25,
    mapper: Optional[Mapper] = None,
    transformer: Optional[RowTransformer] = None,
) -> EvaluationResult:
    """Evaluate retrieval performance on a dataset.
    
    Args:
        dataset: List of query items with 'Question' and 'Correct_rows' keys
        row_strings: List of passage strings to search
        k: Number of results to evaluate at
        mapper: Optional mapper model to transform embeddings
        transformer: Optional transformer (creates new one if None)
        
    Returns:
        EvaluationResult with recall and MRR metrics
    """
    config = get_config()
    device = get_device()
    
    if transformer is None:
        transformer = RowTransformer()
    
    # Build search index
    search_engine = SearchEngine()
    embeddings = transformer.encode_sentence(row_strings)
    
    if mapper is not None:
        mapper.eval()
        with torch.no_grad():
            embeddings = mapper(embeddings)
    
    search_engine.index(embeddings)
    
    # Evaluate
    recall_sum = 0.0
    mrr_sum = 0.0
    
    for item in dataset:
        query_embedding = transformer.encode_sentence(item["Question"]).to(device)
        
        if mapper is not None:
            with torch.no_grad():
                query_embedding = mapper(query_embedding)
        
        _, indices = search_engine.search(query_embedding)
        retrieved = set(indices[0][:k])
        relevant = set(item["Correct_rows"])
        
        recall_sum += compute_recall_at_k(retrieved, relevant)
        mrr_sum += compute_mrr_at_k(list(indices[0][:k]), relevant)
    
    num_queries = len(dataset)
    recall_at_k = (recall_sum / num_queries) * 100
    mrr_at_k = (mrr_sum / num_queries) * 100
    
    return EvaluationResult(recall=recall_at_k, mrr=mrr_at_k, k=k)


def evaluate_multiple_k(
    dataset: List[Dict],
    row_strings: List[str],
    k_values: List[int] = [5, 10, 25, 50],
    mapper: Optional[Mapper] = None,
) -> Dict[int, EvaluationResult]:
    """Evaluate retrieval at multiple K values.
    
    Args:
        dataset: List of query items
        row_strings: List of passage strings
        k_values: List of K values to evaluate
        mapper: Optional mapper model
        
    Returns:
        Dictionary mapping K values to EvaluationResults
    """
    transformer = RowTransformer()
    results = {}
    
    for k in k_values:
        results[k] = evaluate_retrieval(
            dataset=dataset,
            row_strings=row_strings,
            k=k,
            mapper=mapper,
            transformer=transformer,
        )
    
    return results
