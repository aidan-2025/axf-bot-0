#!/usr/bin/env python3
"""
FinBERT Client
Specialized FinBERT client for financial sentiment analysis
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, Pipeline
)

from .models import SentimentResult, SentimentLabel, SentimentSource

logger = logging.getLogger(__name__)

@dataclass
class FinBERTConfig:
    """Configuration for FinBERT client"""
    model_name: str = "yiyanghkust/finbert-tone"
    max_length: int = 512
    batch_size: int = 16
    device: str = "auto"
    confidence_threshold: float = 0.7
    enable_gpu: bool = True
    cache_dir: Optional[str] = None

class FinBERTClient:
    """FinBERT client for financial sentiment analysis"""
    
    def __init__(self, config: FinBERTConfig):
        """Initialize FinBERT client"""
        self.config = config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the FinBERT model"""
        if self._initialized:
            return
        
        try:
            logger.info(f"Initializing FinBERT model: {self.config.model_name}")
            
            # Determine device
            device = self._get_device()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                return_all_scores=True
            )
            
            self._initialized = True
            logger.info("FinBERT model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT model: {e}")
            raise
    
    def _get_device(self) -> int:
        """Determine the best device for inference"""
        if self.config.device == "auto":
            if self.config.enable_gpu and torch.cuda.is_available():
                return 0  # Use first GPU
            else:
                return -1  # Use CPU
        elif self.config.device == "cuda":
            return 0 if torch.cuda.is_available() else -1
        else:
            return -1  # CPU
    
    async def analyze_text(self, 
                          text: str,
                          language: str = "en",
                          currency_pairs: Optional[List[str]] = None) -> SentimentResult:
        """Analyze sentiment of financial text"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Preprocess text
            processed_text = self._preprocess_financial_text(text)
            
            # Run inference
            results = self.pipeline(
                processed_text,
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            )
            
            # Parse results
            sentiment_data = self._parse_finbert_results(results[0])
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create result
            result = SentimentResult(
                text=text,
                label=sentiment_data['label'],
                score=sentiment_data['score'],
                confidence=sentiment_data['confidence'],
                source=SentimentSource.FINBERT,
                language=language,
                processing_time_ms=processing_time,
                model_name=self.config.model_name,
                model_version="1.0",
                currency_pairs=currency_pairs or [],
                financial_entities=self._extract_financial_entities(text),
                market_impact=self._calculate_market_impact(sentiment_data, text)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text with FinBERT: {e}")
            # Return neutral sentiment as fallback
            return SentimentResult(
                text=text,
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.0,
                source=SentimentSource.FINBERT,
                language=language,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_name=self.config.model_name,
                currency_pairs=currency_pairs or []
            )
    
    def _preprocess_financial_text(self, text: str) -> str:
        """Preprocess financial text for FinBERT analysis"""
        # Basic preprocessing
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Financial text specific preprocessing
        # Convert common financial abbreviations
        financial_abbreviations = {
            'GDP': 'gross domestic product',
            'CPI': 'consumer price index',
            'Fed': 'Federal Reserve',
            'ECB': 'European Central Bank',
            'BoE': 'Bank of England',
            'BoJ': 'Bank of Japan',
            'RBA': 'Reserve Bank of Australia',
            'RBNZ': 'Reserve Bank of New Zealand',
            'SNB': 'Swiss National Bank',
            'BoC': 'Bank of Canada'
        }
        
        for abbrev, full_name in financial_abbreviations.items():
            text = text.replace(abbrev, f"{abbrev} ({full_name})")
        
        # Truncate if too long
        if len(text) > self.config.max_length * 4:
            text = text[:self.config.max_length * 4] + "..."
        
        return text
    
    def _parse_finbert_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse FinBERT pipeline results"""
        if not results:
            return {
                'label': SentimentLabel.NEUTRAL,
                'score': 0.0,
                'confidence': 0.0
            }
        
        # Find the highest scoring result
        best_result = max(results, key=lambda x: x['score'])
        
        # Map FinBERT labels to our labels
        label_mapping = {
            'LABEL_0': SentimentLabel.NEGATIVE,
            'LABEL_1': SentimentLabel.NEUTRAL,
            'LABEL_2': SentimentLabel.POSITIVE,
            'NEGATIVE': SentimentLabel.NEGATIVE,
            'NEUTRAL': SentimentLabel.NEUTRAL,
            'POSITIVE': SentimentLabel.POSITIVE
        }
        
        label = label_mapping.get(best_result['label'], SentimentLabel.NEUTRAL)
        score = best_result['score']
        
        # Convert score to -1 to 1 range
        if label == SentimentLabel.NEGATIVE:
            score = -abs(score)
        elif label == SentimentLabel.POSITIVE:
            score = abs(score)
        else:
            score = 0.0
        
        return {
            'label': label,
            'score': score,
            'confidence': score
        }
    
    def _extract_financial_entities(self, text: str) -> List[str]:
        """Extract financial entities from text"""
        entities = []
        
        # Currency pairs
        import re
        currency_pattern = r'\b[A-Z]{3}/[A-Z]{3}\b'
        entities.extend(re.findall(currency_pattern, text.upper()))
        
        # Financial terms
        financial_terms = [
            'GDP', 'CPI', 'inflation', 'interest rate', 'unemployment',
            'retail sales', 'manufacturing', 'trade balance', 'fiscal',
            'monetary policy', 'central bank', 'fed', 'ecb', 'boe',
            'earnings', 'revenue', 'profit', 'loss', 'dividend',
            'bond', 'treasury', 'yield', 'spread', 'volatility'
        ]
        
        text_lower = text.lower()
        for term in financial_terms:
            if term in text_lower:
                entities.append(term.upper())
        
        return list(set(entities))
    
    def _calculate_market_impact(self, sentiment_data: Dict[str, Any], text: str) -> Optional[float]:
        """Calculate market impact based on sentiment and text content"""
        score = sentiment_data['score']
        confidence = sentiment_data['confidence']
        
        # Base impact from sentiment score
        impact = score * confidence
        
        # Adjust for financial keywords
        high_impact_keywords = [
            'rate cut', 'rate hike', 'quantitative easing', 'tapering',
            'recession', 'boom', 'crash', 'rally', 'surge', 'plunge',
            'earnings beat', 'earnings miss', 'guidance', 'forecast'
        ]
        
        text_lower = text.lower()
        for keyword in high_impact_keywords:
            if keyword in text_lower:
                impact *= 1.5  # Amplify impact
                break
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, impact))
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_name": self.config.model_name,
            "device": str(self._get_device()),
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "confidence_threshold": self.config.confidence_threshold
        }
    
    async def close(self):
        """Close the client and cleanup resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
        
        self._initialized = False
        logger.info("FinBERT client closed")

