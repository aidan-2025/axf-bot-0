#!/usr/bin/env python3
"""
BERT Sentiment Analyzer
BERT-based sentiment analysis for financial text
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, Pipeline
)

from .models import SentimentResult, SentimentLabel, SentimentSource

logger = logging.getLogger(__name__)

@dataclass
class BERTConfig:
    """Configuration for BERT sentiment analyzer"""
    model_name: str = "yiyanghkust/finbert-tone"
    max_length: int = 512
    batch_size: int = 16
    device: str = "auto"  # "auto", "cpu", "cuda"
    confidence_threshold: float = 0.7
    enable_gpu: bool = True
    cache_dir: Optional[str] = None

class BERTSentimentAnalyzer:
    """BERT-based sentiment analyzer for financial text"""
    
    def __init__(self, config: BERTConfig):
        """Initialize BERT sentiment analyzer"""
        self.config = config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the BERT model and tokenizer"""
        if self._initialized:
            return
            
        try:
            logger.info(f"Initializing BERT model: {self.config.model_name}")
            
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
            logger.info("BERT model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {e}")
            raise
    
    def _get_device(self) -> Union[int, str]:
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
        """Analyze sentiment of a single text"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Run inference
            results = self.pipeline(
                processed_text,
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            )
            
            # Parse results
            sentiment_data = self._parse_bert_results(results[0])
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create result
            result = SentimentResult(
                text=text,
                label=sentiment_data['label'],
                score=sentiment_data['score'],
                confidence=sentiment_data['confidence'],
                source=SentimentSource.BERT,
                language=language,
                processing_time_ms=processing_time,
                model_name=self.config.model_name,
                currency_pairs=currency_pairs or [],
                financial_entities=self._extract_financial_entities(text)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text with BERT: {e}")
            # Return neutral sentiment as fallback
            return SentimentResult(
                text=text,
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.0,
                source=SentimentSource.BERT,
                language=language,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_name=self.config.model_name,
                currency_pairs=currency_pairs or []
            )
    
    async def analyze_batch(self, 
                           texts: List[str],
                           languages: Optional[List[str]] = None,
                           currency_pairs_list: Optional[List[List[str]]] = None) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        results = []
        
        try:
            # Process in batches
            batch_size = self.config.batch_size
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_languages = (languages or ["en"] * len(texts))[i:i + batch_size]
                batch_currency_pairs = (currency_pairs_list or [[]] * len(texts))[i:i + batch_size]
                
                # Preprocess batch
                processed_texts = [self._preprocess_text(text) for text in batch_texts]
                
                # Run inference
                batch_results = self.pipeline(
                    processed_texts,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=True
                )
                
                # Parse results
                for j, (text, language, currency_pairs, result) in enumerate(
                    zip(batch_texts, batch_languages, batch_currency_pairs, batch_results)
                ):
                    sentiment_data = self._parse_bert_results(result)
                    
                    result_obj = SentimentResult(
                        text=text,
                        label=sentiment_data['label'],
                        score=sentiment_data['score'],
                        confidence=sentiment_data['confidence'],
                        source=SentimentSource.BERT,
                        language=language,
                        processing_time_ms=None,  # Will be calculated for batch
                        model_name=self.config.model_name,
                        currency_pairs=currency_pairs,
                        financial_entities=self._extract_financial_entities(text)
                    )
                    
                    results.append(result_obj)
            
            # Calculate average processing time
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(texts) if texts else 0
            
            for result in results:
                result.processing_time_ms = avg_time
            
            logger.info(f"Analyzed {len(texts)} texts in {total_time:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing batch with BERT: {e}")
            # Return neutral sentiments as fallback
            return [
                SentimentResult(
                    text=text,
                    label=SentimentLabel.NEUTRAL,
                    score=0.0,
                    confidence=0.0,
                    source=SentimentSource.BERT,
                    language=(languages or ["en"] * len(texts))[i],
                    processing_time_ms=(time.time() - start_time) * 1000 / len(texts),
                    model_name=self.config.model_name,
                    currency_pairs=(currency_pairs_list or [[]] * len(texts))[i]
                )
                for i, text in enumerate(texts)
            ]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for BERT analysis"""
        # Basic preprocessing
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > self.config.max_length * 4:  # Rough character to token ratio
            text = text[:self.config.max_length * 4] + "..."
        
        return text
    
    def _parse_bert_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse BERT pipeline results"""
        if not results:
            return {
                'label': SentimentLabel.NEUTRAL,
                'score': 0.0,
                'confidence': 0.0
            }
        
        # Find the highest scoring result
        best_result = max(results, key=lambda x: x['score'])
        
        # Map BERT labels to our labels
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
        
        # Convert score to -1 to 1 range if needed
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
        # Simple entity extraction - in production, use NER model
        entities = []
        
        # Currency pairs
        import re
        currency_pattern = r'\b[A-Z]{3}/[A-Z]{3}\b'
        entities.extend(re.findall(currency_pattern, text.upper()))
        
        # Financial terms
        financial_terms = [
            'GDP', 'CPI', 'inflation', 'interest rate', 'unemployment',
            'retail sales', 'manufacturing', 'trade balance', 'fiscal',
            'monetary policy', 'central bank', 'fed', 'ecb', 'boe'
        ]
        
        text_lower = text.lower()
        for term in financial_terms:
            if term in text_lower:
                entities.append(term.upper())
        
        return list(set(entities))
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_name": self.config.model_name,
            "device": str(self._get_device()),
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size
        }
    
    async def close(self):
        """Close the analyzer and cleanup resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
        
        self._initialized = False
        logger.info("BERT analyzer closed")

