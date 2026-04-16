#!/usr/bin/env python3
"""Main entry point for the Automotive RAG System.

Usage:
    python main.py train           # Run hyperparameter grid search training
    python main.py train --single  # Run single training with default params
    python main.py evaluate        # Evaluate baseline retrieval
    python main.py webapp          # Start the web application
"""

import argparse
import sys

from src.config import load_config, get_config
from src.utils import prepare_data
from src.evaluation import evaluate_retrieval
from src.adaptation import MapperTrainer, TrainingConfig


def train(single: bool = False):
    """Run training for the Mapper model.
    
    Args:
        single: If True, run single training. Otherwise, run grid search.
    """
    print("Loading configuration...")
    config = get_config()
    
    print("Preparing data...")
    row_strings, train_data, test_data = prepare_data()
    print(f"Training set: {len(train_data)} queries")
    print(f"Test set: {len(test_data)} queries")
    print(f"Passages: {len(row_strings)} rows")
    
    # Initialize trainer
    trainer = MapperTrainer(
        row_strings=row_strings,
        train_data=train_data,
        test_data=test_data,
    )
    
    if single:
        # Single training run with default config
        training_config = TrainingConfig()
        print(f"\nRunning single training: {training_config.name}")
        best_mapper, metrics = trainer.train(training_config)
        print("\nFinal metrics:", metrics)
    else:
        # Grid search
        print("\nStarting hyperparameter grid search...")
        results = trainer.grid_search()
        print("\nGrid search complete!")
        print(f"Results saved to: {config.data.results_dir}/mapper_grid.csv")
        print("\nBest configurations by test recall@25:")
        print(results.nlargest(5, "test_recall@25")[
            ["batch_size", "learning_rate", "margin", "test_recall@25", "test_mrr@25"]
        ])


def evaluate():
    """Evaluate baseline retrieval performance."""
    print("Loading configuration...")
    config = get_config()
    
    print("Preparing data...")
    row_strings, train_data, test_data = prepare_data()
    
    print("\nEvaluating baseline retrieval...")
    print("-" * 50)
    
    for k in [5, 10, 25, 50]:
        train_result = evaluate_retrieval(train_data, row_strings, k=k)
        test_result = evaluate_retrieval(test_data, row_strings, k=k)
        print(f"Train {train_result}")
        print(f"Test  {test_result}")
        print()


def webapp():
    """Start the web application."""
    from src.webapp import create_app
    from src.config import get_config
    
    config = get_config()
    app = create_app()
    
    print(f"\nStarting web application on http://{config.webapp.host}:{config.webapp.port}")
    app.run(
        host=config.webapp.host,
        port=config.webapp.port,
        debug=config.webapp.debug,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automotive RAG System - Retrieval Augmented Generation for Car Specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the Mapper model")
    train_parser.add_argument(
        "--single", "-s",
        action="store_true",
        help="Run single training instead of grid search"
    )
    
    # Evaluate command
    subparsers.add_parser("evaluate", help="Evaluate baseline retrieval")
    
    # Webapp command
    subparsers.add_parser("webapp", help="Start the web application")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(single=args.single)
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "webapp":
        webapp()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
