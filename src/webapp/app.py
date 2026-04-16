"""Flask web application for the Automotive RAG System."""

from flask import Flask, render_template, request, redirect, url_for
import numpy as np

from src.config import get_config
from src.models import RowTransformer
from src.retrieval import SearchEngine
from src.generation import Generator
from src.utils import load_gold_data, load_nlq_dataset, create_row_strings, get_sample_queries


# Global state
_app_state = {
    "transformer": None,
    "search_engine": None,
    "row_strings": None,
    "nlq_dataset": None,
    "initialized": False,
}


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    
    @app.before_request
    def ensure_initialized():
        if not _app_state["initialized"]:
            _initialize_app()
    
    @app.route("/")
    def index():
        """Home page with sample queries."""
        config = get_config()
        queries = get_sample_queries(
            _app_state["nlq_dataset"], 
            n=config.webapp.sample_queries
        )
        return render_template("index.html", queries=queries)
    
    @app.route("/process_query", methods=["GET", "POST"])
    def process_query():
        """Process a query and return results."""
        if request.method == "GET":
            index = int(request.args.get("index", -1))
            if index >= 0:
                query_text = _app_state["nlq_dataset"][index]["Question"]
                correct_rows = _app_state["nlq_dataset"][index]["Correct_rows"]
            else:
                return redirect(url_for("index"))
        else:
            query_text = request.form.get("customQuery", "")
            correct_rows = []
        
        if not query_text:
            return redirect(url_for("index"))
        
        # Encode query and search
        query_embedding = _app_state["transformer"].encode_sentence(query_text)
        query_np = query_embedding.cpu().numpy().reshape(1, -1)
        _, indices = _app_state["search_engine"].search(query_embedding)
        
        # Prepare results
        retrieved_items = [
            {"index": int(idx), "text": _app_state["row_strings"][int(idx)]}
            for idx in indices[0][:5]
        ]
        
        related_items = [
            {"index": idx, "text": _app_state["row_strings"][idx]}
            for idx in correct_rows
        ]
        
        # Generate LLM response
        context_rows = [_app_state["row_strings"][int(idx)] for idx in indices[0][:5]]
        
        try:
            generator = Generator()
            llm_response = generator.generate_with_rows(query_text, context_rows)
        except Exception as e:
            llm_response = f"Error generating response: {str(e)}"
        
        return render_template(
            "results.html",
            query_text=query_text,
            retrieved_items=retrieved_items,
            related_items=related_items,
            llm_response=llm_response,
        )
    
    return app


def _initialize_app():
    """Initialize the application state."""
    print("Initializing application...")
    
    # Load data
    gold_df = load_gold_data()
    _app_state["row_strings"] = create_row_strings(gold_df)
    _app_state["nlq_dataset"] = load_nlq_dataset()
    
    # Initialize transformer
    print("Loading transformer model...")
    _app_state["transformer"] = RowTransformer()
    
    # Encode passages and build index
    print("Encoding passages...")
    embeddings = _app_state["transformer"].encode_batch(
        _app_state["row_strings"],
        show_progress=True
    )
    
    print("Building search index...")
    _app_state["search_engine"] = SearchEngine()
    _app_state["search_engine"].index(embeddings)
    
    _app_state["initialized"] = True
    print("Application initialized successfully!")


def run_app():
    """Run the Flask application."""
    config = get_config()
    app = create_app()
    app.run(
        host=config.webapp.host,
        port=config.webapp.port,
        debug=config.webapp.debug,
    )


if __name__ == "__main__":
    run_app()
