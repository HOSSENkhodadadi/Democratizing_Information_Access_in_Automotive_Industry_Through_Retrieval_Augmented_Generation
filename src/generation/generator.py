"""LLM-based response generator using Groq API."""

import os
from typing import Optional

from groq import Groq

from src.config import get_config


class Generator:
    """LLM response generator using Groq API.
    
    Generates responses to user queries based on retrieved context
    using large language models.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize the generator.
        
        Args:
            api_key: Groq API key. Uses GROQ_API_KEY env var if None.
            model: Model name to use. Uses config if None.
            system_prompt: System prompt for the LLM. Uses config if None.
        """
        config = get_config()
        
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model or config.generation.model
        self.system_prompt = system_prompt or config.generation.system_prompt
        self.temperature = config.generation.temperature
        self.top_p = config.generation.top_p
        
        self.client = Groq(api_key=self.api_key)

    def generate(self, query: str, context: str) -> str:
        """Generate a response to the query using the provided context.
        
        Args:
            query: User's question
            context: Retrieved context to base the answer on
            
        Returns:
            Generated response string
        """
        prompt = self._build_prompt(query, context)
        return self._call_api(prompt)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the complete prompt for the LLM."""
        return f"{self.system_prompt}\n\nQuery: {query}\n\nContext:\n{context}"
    
    def _call_api(self, prompt: str) -> str:
        """Call the Groq API and return the response.
        
        Args:
            prompt: Complete prompt to send to the model
            
        Returns:
            Generated response text
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
            stop=None
        )
        
        response_text = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                response_text += content
        
        return response_text

    def generate_with_rows(
        self, 
        query: str, 
        rows: list,
        row_limit: int = 5
    ) -> str:
        """Generate a response using row data as context.
        
        Args:
            query: User's question
            rows: List of row strings to use as context
            row_limit: Maximum number of rows to include
            
        Returns:
            Generated response string
        """
        context_parts = [
            f"{i}: {row}" 
            for i, row in enumerate(rows[:row_limit])
        ]
        context = ";\n".join(context_parts)
        return self.generate(query, context)
