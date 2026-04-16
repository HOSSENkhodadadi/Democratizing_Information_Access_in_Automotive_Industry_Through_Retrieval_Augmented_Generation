# Automotive RAG System

A Retrieval Augmented Generation (RAG) system for democratizing information access in the automotive industry. This project enables semantic search over car specification data and generates natural language responses using LLMs.

## Features

- **Semantic Search**: FAISS-based vector search with sentence transformers
- **Mapper Adaptation**: Train a linear mapping layer to improve retrieval accuracy
- **LLM Generation**: Generate responses using Groq API (Llama 3.3)
- **Web Interface**: Flask-based Q&A interface

## Project Structure

```
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   └── raw/                  # Raw data files
│       ├── gold_initial.csv
│       └── query_dataset_10_255_main.json
├── src/
│   ├── adaptation/           # Mapper training module
│   ├── evaluation/           # Metrics and evaluation
│   ├── generation/           # LLM response generation
│   ├── models/               # Mapper and Transformer models
│   ├── retrieval/            # FAISS search engine
│   ├── utils/                # Data utilities
│   └── webapp/               # Flask web application
├── results/                  # Training results and logs
├── main.py                   # CLI entry point
├── pyproject.toml            # Project dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/automotive-rag.git
cd automotive-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Configuration

Edit `config/config.yaml` to customize settings:

- Model selection and embedding dimensions
- Training hyperparameters
- Web app host/port
- Data paths

## Usage

### Evaluate Baseline Retrieval

```bash
python main.py evaluate
```

### Train Mapper Model

```bash
# Single training run
python main.py train --single

# Hyperparameter grid search
python main.py train
```

### Run Web Application

```bash
# Set your Groq API key
export GROQ_API_KEY="your-api-key"

# Start the web app
python main.py webapp
```

Then open http://localhost:5005 in your browser.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | API key for Groq LLM service |

## License

MIT License
