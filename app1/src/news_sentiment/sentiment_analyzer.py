#!/usr/bin/env python3
"""
Sentiment Analyzer
BERT-based sentiment analysis for financial news and social media
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import re

from .models import SentimentData, SentimentLabel, NewsSource

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """BERT-based sentiment analyzer for financial text"""
    
    def __init__(self, 
                 model_name: str = "ProsusAI/finbert",
                 max_length: int = 512,
                 batch_size: int = 32):
        """Initialize sentiment analyzer"""
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Placeholder for model loading
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        # Language detection patterns
        self.language_patterns = {
            'en': r'[a-zA-Z]',
            'es': r'[ñáéíóúü]',
            'fr': r'[àâäéèêëïîôöùûüÿç]',
            'de': r'[äöüß]',
            'it': r'[àèéìíîòóùú]',
            'pt': r'[ãõáéíóúâêô]',
            'ru': r'[а-яё]',
            'zh': r'[\u4e00-\u9fff]',
            'ja': r'[\u3040-\u309f\u30a0-\u30ff]'
        }
        
        # Financial sentiment keywords
        self.positive_keywords = {
            'en': ['bullish', 'positive', 'strong', 'growth', 'rise', 'gain', 'up', 'optimistic', 'recovery', 'surge'],
            'es': ['alcista', 'positivo', 'fuerte', 'crecimiento', 'subida', 'ganancia', 'optimista', 'recuperación'],
            'fr': ['haussier', 'positif', 'fort', 'croissance', 'hausse', 'gain', 'optimiste', 'récupération'],
            'de': ['bullisch', 'positiv', 'stark', 'wachstum', 'anstieg', 'gewinn', 'optimistisch', 'erholung'],
            'it': ['rialzista', 'positivo', 'forte', 'crescita', 'aumento', 'guadagno', 'ottimista', 'recupero'],
            'pt': ['altista', 'positivo', 'forte', 'crescimento', 'alta', 'ganho', 'otimista', 'recuperação'],
            'ru': ['бычий', 'позитивный', 'сильный', 'рост', 'подъем', 'прибыль', 'оптимистичный', 'восстановление'],
            'zh': ['看涨', '积极', '强劲', '增长', '上涨', '收益', '乐观', '复苏'],
            'ja': ['強気', 'ポジティブ', '強い', '成長', '上昇', '利益', '楽観的', '回復']
        }
        
        self.negative_keywords = {
            'en': ['bearish', 'negative', 'weak', 'decline', 'fall', 'loss', 'down', 'pessimistic', 'crash', 'plunge'],
            'es': ['bajista', 'negativo', 'débil', 'declive', 'caída', 'pérdida', 'pesimista', 'colapso'],
            'fr': ['baissier', 'négatif', 'faible', 'déclin', 'chute', 'perte', 'pessimiste', 'effondrement'],
            'de': ['bärisch', 'negativ', 'schwach', 'rückgang', 'fall', 'verlust', 'pessimistisch', 'crash'],
            'it': ['ribassista', 'negativo', 'debole', 'declino', 'caduta', 'perdita', 'pessimista', 'crollo'],
            'pt': ['baixista', 'negativo', 'fraco', 'declínio', 'queda', 'perda', 'pessimista', 'colapso'],
            'ru': ['медвежий', 'негативный', 'слабый', 'снижение', 'падение', 'убыток', 'пессимистичный', 'крах'],
            'zh': ['看跌', '消极', '疲软', '下降', '下跌', '亏损', '悲观', '崩盘'],
            'ja': ['弱気', 'ネガティブ', '弱い', '下降', '下落', '損失', '悲観的', '暴落']
        }
    
    async def initialize(self):
        """Initialize the sentiment analyzer (load models)"""
        if self._model_loaded:
            return
        
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # TODO: Implement actual model loading
            # For now, we'll use a placeholder implementation
            self._model_loaded = True
            
            logger.info("Sentiment analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            raise
    
    async def analyze_text(self, 
                          text: str,
                          language: str = 'en',
                          source: Optional[NewsSource] = None) -> Optional[SentimentData]:
        """Analyze sentiment of a single text"""
        if not self._model_loaded:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Detect language if not provided
            if not language or language == 'auto':
                language = self._detect_language(text)
            
            # Preprocess text
            processed_text = self._preprocess_text(text, language)
            
            # Analyze sentiment
            sentiment_result = await self._analyze_sentiment(processed_text, language)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create sentiment data
            sentiment_id = f"sent_{hashlib.md5(f'{text}{language}'.encode()).hexdigest()[:12]}"
            
            sentiment_data = SentimentData(
                sentiment_id=sentiment_id,
                text=text,
                language=language,
                label=sentiment_result['label'],
                score=sentiment_result['score'],
                confidence=sentiment_result['confidence'],
                model_name=self.model_name,
                model_version="1.0",
                source=source or NewsSource.TWITTER,
                processed_at=datetime.utcnow(),
                processing_time_ms=processing_time
            )
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None
    
    async def analyze_batch(self, 
                           texts: List[str],
                           languages: Optional[List[str]] = None,
                           sources: Optional[List[NewsSource]] = None) -> List[SentimentData]:
        """Analyze sentiment of multiple texts in batch"""
        if not self._model_loaded:
            await self.initialize()
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_languages = (languages[i:i + self.batch_size] if languages else ['en'] * len(batch_texts))
            batch_sources = (sources[i:i + self.batch_size] if sources else [NewsSource.TWITTER] * len(batch_texts))
            
            # Process batch
            batch_results = await self._analyze_batch_sentiment(
                batch_texts, batch_languages, batch_sources
            )
            
            results.extend(batch_results)
        
        return results
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        text_lower = text.lower()
        
        # Check each language pattern
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, text_lower):
                return lang
        
        # Default to English
        return 'en'
    
    def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (for social media)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    async def _analyze_sentiment(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze sentiment using the model"""
        # TODO: Implement actual BERT model inference
        # For now, use rule-based sentiment analysis
        
        # Get language-specific keywords
        positive_keywords = self.positive_keywords.get(language, self.positive_keywords['en'])
        negative_keywords = self.negative_keywords.get(language, self.negative_keywords['en'])
        
        text_lower = text.lower()
        
        # Count positive and negative keywords
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        # Calculate sentiment score
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            score = 0.0
            label = SentimentLabel.NEUTRAL
            confidence = 0.5
        else:
            score = (positive_count - negative_count) / total_keywords
            confidence = min(total_keywords / 10.0, 1.0)  # More keywords = higher confidence
            
            if score > 0.3:
                label = SentimentLabel.POSITIVE
            elif score < -0.3:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL
        
        return {
            'label': label,
            'score': score,
            'confidence': confidence
        }
    
    async def _analyze_batch_sentiment(self, 
                                     texts: List[str],
                                     languages: List[str],
                                     sources: List[NewsSource]) -> List[SentimentData]:
        """Analyze sentiment for a batch of texts"""
        results = []
        
        for i, text in enumerate(texts):
            try:
                sentiment_data = await self.analyze_text(
                    text=text,
                    language=languages[i],
                    source=sources[i]
                )
                
                if sentiment_data:
                    results.append(sentiment_data)
                    
            except Exception as e:
                logger.error(f"Error analyzing sentiment for text {i}: {e}")
                continue
        
        return results
    
    async def get_sentiment_summary(self, 
                                  currency_pairs: List[str],
                                  hours: int = 24) -> Dict[str, Any]:
        """Get sentiment summary for currency pairs"""
        # TODO: Implement actual sentiment aggregation from database
        logger.info(f"Getting sentiment summary for {currency_pairs} over {hours} hours")
        
        # Placeholder implementation
        return {
            'currency_pairs': currency_pairs,
            'time_period_hours': hours,
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.5,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'total_articles': 0,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    async def get_trending_sentiment(self, 
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending sentiment topics"""
        # TODO: Implement actual trending analysis from database
        logger.info(f"Getting trending sentiment topics (limit={limit})")
        
        # Placeholder implementation
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the sentiment model"""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'model_loaded': self._model_loaded,
            'supported_languages': list(self.language_patterns.keys()),
            'positive_keywords_count': sum(len(keywords) for keywords in self.positive_keywords.values()),
            'negative_keywords_count': sum(len(keywords) for keywords in self.negative_keywords.values())
        }
    
    async def close(self):
        """Close the sentiment analyzer and cleanup resources"""
        if self._model_loaded:
            # TODO: Implement model cleanup
            self._model_loaded = False
            logger.info("Sentiment analyzer closed")

