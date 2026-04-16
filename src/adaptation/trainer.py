"""Training logic for the Mapper model."""

import os
import copy
import random
import gc
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import get_config, get_device
from src.models import Mapper, RowTransformer
from src.retrieval import SearchEngine
from src.evaluation import evaluate_retrieval


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    positive_tendency: float = 0.9
    margin: float = 0.5
    regularization_strength: float = 0.1
    
    @property
    def name(self) -> str:
        """Generate a unique name for this config."""
        return (
            f"bs{self.batch_size}_lr{self.learning_rate}_"
            f"reg{self.regularization_strength}_margin{self.margin}_"
            f"pt{self.positive_tendency}"
        )


class MapperTrainer:
    """Trainer for the Mapper model."""
    
    def __init__(
        self,
        row_strings: List[str],
        train_data: List[Dict],
        test_data: List[Dict],
        results_dir: str = None,
    ):
        """Initialize the trainer.
        
        Args:
            row_strings: List of passage strings
            train_data: Training dataset
            test_data: Test dataset
            results_dir: Directory to save results. Uses config if None.
        """
        self.config = get_config()
        self.device = get_device()
        
        self.row_strings = row_strings
        self.train_data = train_data
        self.test_data = test_data
        
        self.results_dir = results_dir or str(
            self.config.get_absolute_path(self.config.data.results_dir)
        )
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "mapper_logs"), exist_ok=True)
        
        # Initialize transformer and compute passage embeddings
        self.transformer = RowTransformer()
        self.passage_embeddings = (
            self.transformer.encode_sentence(row_strings).cpu().numpy()
        )
        
        # Compute baseline retrieved indices
        self._compute_baseline_indices()
    
    def _compute_baseline_indices(self):
        """Compute baseline retrieval indices for training data."""
        search_engine = SearchEngine()
        search_engine.index(torch.tensor(self.passage_embeddings))
        
        self.baseline_indices = []
        for item in self.train_data:
            query_embedding = self.transformer.encode_sentence(item["Question"])
            _, indices = search_engine.search(query_embedding)
            self.baseline_indices.append(indices[0])
        self.baseline_indices = np.array(self.baseline_indices)
    
    def train(
        self,
        training_config: TrainingConfig,
        epochs: int = None,
        patience: int = None,
    ) -> Tuple[Mapper, Dict]:
        """Train the mapper with given configuration.
        
        Args:
            training_config: Training hyperparameters
            epochs: Number of epochs. Uses global config if None.
            patience: Early stopping patience. Uses global config if None.
            
        Returns:
            Tuple of (best_mapper, final_metrics)
        """
        epochs = epochs or self.config.training.epochs
        patience = patience or self.config.training.patience
        
        # Initialize mapper and optimizer
        mapper = Mapper().to(self.device)
        optimizer = torch.optim.Adam(mapper.parameters(), lr=training_config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        
        best_mapper = mapper
        best_test_recall = -float("inf")
        epochs_since_improvement = 0
        
        log_path = os.path.join(
            self.results_dir, "mapper_logs", f"{training_config.name}.csv"
        )
        
        for epoch in range(epochs):
            loss_list = self._train_epoch(
                mapper, optimizer, training_config
            )
            avg_loss = np.mean(loss_list) if loss_list else float("nan")
            
            # Evaluate
            mapper.eval()
            with torch.no_grad():
                train_result = evaluate_retrieval(
                    self.train_data, self.row_strings, k=25, mapper=mapper
                )
                test_result = evaluate_retrieval(
                    self.test_data, self.row_strings, k=25, mapper=mapper
                )
                
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1:02}/{epochs}, Loss: {avg_loss:.4f}, "
                    f"Test: {test_result}, Train: {train_result}, LR: {current_lr:.6f}"
                )
                
                # Log results
                self._log_epoch(
                    log_path, epoch + 1, avg_loss, 
                    train_result.recall, test_result.recall,
                    train_result.mrr, test_result.mrr
                )
                
                # Update scheduler
                scheduler.step(test_result.recall)
                
                # Check for improvement
                if test_result.recall > best_test_recall:
                    best_mapper = copy.deepcopy(mapper)
                    best_test_recall = test_result.recall
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                
                if epochs_since_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Final evaluation
        final_metrics = self._final_evaluation(best_mapper)
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        return best_mapper, final_metrics
    
    def _train_epoch(
        self,
        mapper: Mapper,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
    ) -> List[float]:
        """Train for one epoch."""
        mapper.train()
        loss_list = []
        
        query_indices = list(range(len(self.train_data)))
        random.shuffle(query_indices)
        
        with tqdm(total=len(self.train_data) // config.batch_size + 1) as pbar:
            for step in range(0, len(self.train_data), config.batch_size):
                batch_indices = query_indices[step:step + config.batch_size]
                if not batch_indices:
                    continue
                
                # Get batch embeddings
                batch_questions = [
                    self.train_data[i]["Question"] for i in batch_indices
                ]
                batch_query_embeddings = self.transformer.encode_sentence(
                    batch_questions
                ).to(self.device)
                batch_mapped = mapper(batch_query_embeddings)
                
                # Get targets
                batch_pos, batch_neg = self._get_targets(
                    batch_indices, config.positive_tendency
                )
                
                # Compute loss
                loss = self._compute_loss(
                    batch_mapped, batch_pos, batch_neg,
                    config.margin, config.regularization_strength, mapper
                )
                
                if loss is not None and not torch.isnan(loss):
                    optimizer.zero_grad()
                    torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)
                    loss.backward()
                    optimizer.step()
                    loss_list.append(loss.item())
                
                pbar.set_postfix_str(
                    f"loss: {np.mean(loss_list) if loss_list else float('nan'):.4f}",
                    refresh=False
                )
                pbar.update(1)
        
        return loss_list
    
    def _get_targets(
        self,
        batch_indices: List[int],
        positive_tendency: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get positive and negative target embeddings for batch."""
        preferred_total = self.config.training.preferred_total
        embedding_dim = self.config.model.embedding_dim
        
        batch_positives_counts = []
        batch_negatives_counts = []
        
        for idx in batch_indices:
            correct_rows = self.train_data[idx]["Correct_rows"]
            total_pos = max(1, min(
                round(preferred_total * positive_tendency),
                len(correct_rows)
            ))
            total_neg = max(1, min(
                round(preferred_total * (1 - positive_tendency)),
                len(self.row_strings) - len(correct_rows)
            ))
            batch_positives_counts.append(total_pos)
            batch_negatives_counts.append(total_neg)
        
        max_pos = max(batch_positives_counts)
        max_neg = max(batch_negatives_counts)
        
        batch_pos = torch.full(
            (len(batch_indices), max_pos, embedding_dim),
            float("nan"), device=self.device
        )
        batch_neg = torch.full(
            (len(batch_indices), max_neg, embedding_dim),
            float("nan"), device=self.device
        )
        
        for i, (idx, n_pos, n_neg) in enumerate(
            zip(batch_indices, batch_positives_counts, batch_negatives_counts)
        ):
            correct_rows = set(self.train_data[idx]["Correct_rows"])
            all_indices = list(range(len(self.row_strings)))
            incorrect = [j for j in all_indices if j not in correct_rows]
            
            pos_indices = random.sample(list(correct_rows), min(n_pos, len(correct_rows)))
            neg_indices = random.sample(incorrect, min(n_neg, len(incorrect)))
            
            if pos_indices:
                batch_pos[i, :len(pos_indices), :] = torch.tensor(
                    self.passage_embeddings[pos_indices], device=self.device
                )
            if neg_indices:
                batch_neg[i, :len(neg_indices), :] = torch.tensor(
                    self.passage_embeddings[neg_indices], device=self.device
                )
        
        return batch_pos, batch_neg
    
    def _compute_loss(
        self,
        mapped_queries: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
        margin: float,
        reg_strength: float,
        mapper: Mapper,
    ) -> Optional[torch.Tensor]:
        """Compute contrastive loss with regularization."""
        embedding_dim = self.config.model.embedding_dim
        norm_order = self.config.training.norm_order
        
        agg_pos = torch.nanmean(positive_embeddings, dim=1)
        agg_neg = torch.nanmean(negative_embeddings, dim=1)
        
        epsilon = 1e-8
        pos_scores = torch.norm(mapped_queries - agg_pos + epsilon, p=norm_order, dim=1)
        neg_scores = torch.norm(mapped_queries - agg_neg + epsilon, p=norm_order, dim=1)
        
        pos_loss = torch.nanmean(pos_scores ** 2)
        neg_loss = torch.nanmean(torch.relu(margin - neg_scores) ** 2)
        
        # L2 regularization towards identity
        identity = 0.9 * torch.eye(embedding_dim).to(self.device)
        l2_loss = reg_strength * torch.norm(mapper.linear.weight - identity, p=2)
        
        if not torch.isnan(pos_loss) and not torch.isnan(neg_loss):
            return pos_loss + neg_loss + l2_loss
        elif not torch.isnan(pos_loss):
            return pos_loss + l2_loss
        elif not torch.isnan(neg_loss):
            return neg_loss + l2_loss
        return None
    
    def _log_epoch(
        self, path: str, epoch: int, loss: float,
        train_recall: float, test_recall: float,
        train_mrr: float, test_mrr: float
    ):
        """Log epoch results to CSV."""
        row = {
            "epoch": epoch,
            "loss": loss,
            "train_recall@25": train_recall,
            "test_recall@25": test_recall,
            "train_mrr@25": train_mrr,
            "test_mrr@25": test_mrr,
        }
        df = pd.DataFrame([row])
        
        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            existing = pd.read_csv(path)
            pd.concat([existing, df], ignore_index=True).to_csv(path, index=False)
    
    def _final_evaluation(self, mapper: Mapper) -> Dict:
        """Run final evaluation at multiple K values."""
        results = {}
        
        for k in [25, 50]:
            train_result = evaluate_retrieval(
                self.train_data, self.row_strings, k=k, mapper=mapper
            )
            test_result = evaluate_retrieval(
                self.test_data, self.row_strings, k=k, mapper=mapper
            )
            
            results[f"train_recall@{k}"] = train_result.recall
            results[f"train_mrr@{k}"] = train_result.mrr
            results[f"test_recall@{k}"] = test_result.recall
            results[f"test_mrr@{k}"] = test_result.mrr
            
            print(f"Final Train @{k}: {train_result}")
            print(f"Final Test @{k}: {test_result}")
        
        return results
    
    def grid_search(self) -> pd.DataFrame:
        """Run grid search over all hyperparameter combinations.
        
        Returns:
            DataFrame with all results
        """
        hp = self.config.hyperparameters
        results = []
        
        for batch_size in hp.batch_sizes:
            for lr in hp.learning_rates:
                for pt in hp.positive_tendencies:
                    for margin in hp.margins:
                        for reg in hp.regularization_strengths:
                            config = TrainingConfig(
                                batch_size=batch_size,
                                learning_rate=lr,
                                positive_tendency=pt,
                                margin=margin,
                                regularization_strength=reg,
                            )
                            
                            print(f"\n{'='*50}")
                            print(f"Training: {config.name}")
                            print(f"{'='*50}")
                            
                            _, metrics = self.train(config)
                            
                            row = {
                                "batch_size": batch_size,
                                "learning_rate": lr,
                                "positive_tendency": pt,
                                "margin": margin,
                                "regularization": reg,
                                **metrics,
                            }
                            results.append(row)
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_dir, "mapper_grid.csv"), index=False)
        
        return df
