"""Data loading and preprocessing utilities."""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_config


def load_gold_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load the gold standard car specifications data.
    
    Args:
        path: Path to CSV file. Uses config if None.
        
    Returns:
        DataFrame with car specifications
    """
    config = get_config()
    if path is None:
        path = config.get_absolute_path(config.data.gold_csv)
    return pd.read_csv(path)


def load_nlq_dataset(path: Optional[str] = None) -> List[Dict]:
    """Load the natural language query dataset.
    
    Args:
        path: Path to JSON file. Uses config if None.
        
    Returns:
        List of query dictionaries with 'Question' and 'Correct_rows' keys
    """
    config = get_config()
    if path is None:
        path = config.get_absolute_path(config.data.nlq_dataset)
    
    with open(path, "r") as f:
        return json.load(f)


def preprocess_row(row: pd.Series, columns: List[str]) -> str:
    """Convert a DataFrame row to a natural language string.
    
    Args:
        row: DataFrame row
        columns: List of column names to include
        
    Returns:
        Natural language description of the row
    """
    parts = []
    for col in columns:
        if col != "Unnamed: 0" and pd.notnull(row.get(col)):
            parts.append(f"{col} is {row[col]}")
    return "For this car the " + ", ".join(parts)


def create_row_strings(df: pd.DataFrame) -> List[str]:
    """Create natural language strings from all DataFrame rows.
    
    Args:
        df: DataFrame with car specifications
        
    Returns:
        List of natural language strings, one per row
    """
    columns = df.columns.tolist()
    return df.apply(lambda row: preprocess_row(row, columns), axis=1).tolist()


def split_dataset(
    dataset: List[Dict],
    test_size: float = None,
    random_state: int = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and test sets.
    
    Args:
        dataset: Full dataset to split
        test_size: Fraction for test set. Uses config if None.
        random_state: Random seed. Uses config if None.
        
    Returns:
        Tuple of (train_data, test_data)
    """
    config = get_config()
    test_size = test_size or config.training.test_size
    random_state = random_state or config.training.random_state
    
    train_data, test_data = train_test_split(
        dataset, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True
    )
    return train_data, test_data


def prepare_data() -> Tuple[List[str], List[Dict], List[Dict]]:
    """Load and prepare all data for training/evaluation.
    
    Returns:
        Tuple of (row_strings, train_data, test_data)
    """
    # Load gold data and create row strings
    gold_df = load_gold_data()
    row_strings = create_row_strings(gold_df)
    
    # Load and split NLQ dataset
    nlq_dataset = load_nlq_dataset()
    train_data, test_data = split_dataset(nlq_dataset)
    
    return row_strings, train_data, test_data


def get_sample_queries(dataset: List[Dict], n: int = 5) -> List[Dict]:
    """Get random sample of queries from dataset.
    
    Args:
        dataset: Full query dataset
        n: Number of samples to return
        
    Returns:
        List of sampled query dictionaries
    """
    import random
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    return [
        {"index": idx, "Question": dataset[idx]["Question"]}
        for idx in indices
    ]
