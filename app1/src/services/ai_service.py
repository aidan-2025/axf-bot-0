#!/usr/bin/env python3
"""
AI Service
Provider-agnostic AI client with role-based routing:
- research -> Perplexity Sonar-Pro
- generation -> OpenAI (gpt-4o-mini by default)
"""

import os
import logging
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv

# Load environment variables as fallback
load_dotenv("../../.env")

logger = logging.getLogger(__name__)


class AIService:
    """Simple AI service to perform research and generation tasks"""

    def __init__(self) -> None:
        # Load environment variables in constructor
        import pathlib
        env_path = pathlib.Path(__file__).parent.parent.parent.parent / ".env"
        load_dotenv(env_path)
        
        self.research_provider = os.getenv("AI_PROVIDER_RESEARCH", "perplexity").lower()
        self.generation_provider = os.getenv("AI_PROVIDER_GENERATION", "openai").lower()

        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", "")
        self.perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar-medium-online")

        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model_code = os.getenv("OPENAI_MODEL_CODE", "gpt-4o-mini")
        
        # Debug logging
        print(f"AIService initialized - PERPLEXITY_API_KEY: {bool(self.perplexity_api_key)}, OPENAI_API_KEY: {bool(self.openai_api_key)}")
        logger.info(f"AIService initialized - PERPLEXITY_API_KEY: {bool(self.perplexity_api_key)}, OPENAI_API_KEY: {bool(self.openai_api_key)}")

    def research(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Run web-grounded research via Perplexity Sonar-Pro and return citations."""
        if self.research_provider != "perplexity":
            raise ValueError("Only Perplexity is supported for research in this setup")
        if not self.perplexity_api_key:
            raise ValueError("PERPLEXITY_API_KEY is not set")

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json",
        }
        messages: List[Dict[str, str]] = [{"role": "user", "content": query}]
        if context:
            messages.insert(0, {"role": "system", "content": context})

        payload = {
            "model": self.perplexity_model,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.9,
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "stream": False,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", []) or data.get("sources", [])
        return {"content": content, "citations": citations, "raw": data}

    def generate_strategy(self, prompt: str, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use OpenAI to generate a strategy description or reasoning output."""
        if self.generation_provider != "openai":
            raise ValueError("Only OpenAI is supported for generation in this setup")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are an expert quantitative FX strategist. Write clear, structured outputs."},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.openai_model_code,
            "messages": messages,
            "temperature": 0.3,
        }
        if constraints:
            payload["metadata"] = constraints

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"content": content, "raw": data}


